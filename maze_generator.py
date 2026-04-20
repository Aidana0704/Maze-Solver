"""
Programmatic maze environment and generator.

GridMazeEnvironment implements the same MazeEnvironment interface as
AStarMazeEnvironment but is backed by a procedurally-generated graph
rather than a PNG image.

The maze is a list[list[MazeCell]] — the same type produced by
load_image.load_image_into_graph().  Every cell is a node; a wall between
two adjacent cells is simply a missing edge (the neighbour attribute is None).

generate_maze() builds this graph with iterative randomised DFS
(recursive-backtracker), then overlays fire hazards, teleporters, and
confusion cells.
"""
from __future__ import annotations

from collections import deque
from typing import override

import numpy as np

from load_image import CellState, MazeCell
from maze_solver import Action, MazeEnvironment, TurnResult

# ── Direction helpers ────────────────────────────────────────────────────────
# (dr, dc, forward_attr, backward_attr)
_CELL_DIRS: list[tuple[int, int, str, str]] = [
    (-1,  0, 'top_square',    'bottom_square'),
    ( 0,  1, 'right_square',  'left_square'),
    ( 1,  0, 'bottom_square', 'top_square'),
    ( 0, -1, 'left_square',   'right_square'),
]

_ACTION_TO_NEIGHBOR: dict[Action, str] = {
    Action.MOVE_UP:    'top_square',
    Action.MOVE_DOWN:  'bottom_square',
    Action.MOVE_LEFT:  'left_square',
    Action.MOVE_RIGHT: 'right_square',
}

_CONFUSION_FLIP: dict[Action, Action] = {
    Action.MOVE_UP:    Action.MOVE_DOWN,
    Action.MOVE_DOWN:  Action.MOVE_UP,
    Action.MOVE_LEFT:  Action.MOVE_RIGHT,
    Action.MOVE_RIGHT: Action.MOVE_LEFT,
    Action.WAIT:       Action.WAIT,
}

# Fire V-shape arm offsets (dx, dy) from center at rotation 0 (UP×2, RIGHT×2).
# Rotation r: apply (dx,dy) → (-dy, dx) r times (90° CW, same as AStarMazeEnvironment).
_FIRE_ARM_OFFSETS: list[tuple[int, int]] = [(0, -1), (0, -2), (1, 0), (2, 0)]

_TELEPORT_STATES = {CellState.PURPLE_TELEPORT, CellState.GREEN_TELEPORT, CellState.YELLOW_TELEPORT}


class GridMazeEnvironment(MazeEnvironment):
    """
    Graph-based maze environment that mirrors AStarMazeEnvironment's interface.
    Accepts a list[list[MazeCell]] produced by generate_maze().
    Walls are absent edges between adjacent nodes — not grid cells.
    """

    def __init__(self, graph: list[list[MazeCell]]):
        self._graph = graph
        self._rows  = len(graph)
        self._cols  = len(graph[0]) if self._rows > 0 else 0

        start_cell = goal_cell = None
        for row in graph:
            for cell in row:
                if cell.state == CellState.START:
                    start_cell = cell
                elif cell.state == CellState.GOAL:
                    goal_cell = cell
        if start_cell is None or goal_cell is None:
            raise ValueError("Graph must contain exactly one START and one GOAL cell.")

        self._start_cell = start_cell
        self._goal_cell  = goal_cell
        self._goal_pos: tuple[int, int] = goal_cell.position

        # Teleport map: position → destination MazeCell
        self._teleport_map: dict[tuple[int, int], MazeCell] = {}
        pairs: dict[CellState, list[MazeCell]] = {}
        for row in graph:
            for cell in row:
                if cell.state in _TELEPORT_STATES:
                    pairs.setdefault(cell.state, []).append(cell)
        for cells in pairs.values():
            if len(cells) >= 2:
                a, b = cells[0], cells[1]
                self._teleport_map[a.position] = b
                self._teleport_map[b.position] = a

        # Fire centers persist across episodes; _fire_turn never resets
        self._fire_centers: list[tuple[int, int]] = [
            cell.position
            for row in graph for cell in row
            if cell.state == CellState.HAZARD
        ]
        self._fire_turn: int = 0

        # Episode state
        self._current_cell:    MazeCell = self._start_cell
        self._confusion_steps: int = 0
        self._turns:           int = 0
        self._deaths:          int = 0
        self._confused_count:  int = 0
        self._explored:        set[tuple[int, int]] = set()
        self._goal_reached:    bool = False

    # ── Fire helpers ─────────────────────────────────────────────────────────

    def _active_fire_cells(self) -> set[tuple[int, int]]:
        """(x,y) positions currently on fire — mirrors AStarMazeEnvironment logic."""
        rot = self._fire_turn % 4
        active: set[tuple[int, int]] = set()
        for cx, cy in self._fire_centers:
            active.add((cx, cy))
            for dx, dy in _FIRE_ARM_OFFSETS:
                rdx, rdy = dx, dy
                for _ in range(rot):
                    rdx, rdy = -rdy, rdx  # 90° CW
                px, py = cx + rdx, cy + rdy
                if 0 <= px < self._cols and 0 <= py < self._rows:
                    active.add((px, py))
        return active

    # ── MazeEnvironment interface ─────────────────────────────────────────────

    @override
    def reset(self) -> tuple[int, int]:
        self._current_cell    = self._start_cell
        self._confusion_steps = 0
        self._turns           = 0
        self._deaths          = 0
        self._confused_count  = 0
        self._explored        = {self._start_cell.position}
        self._goal_reached    = False
        return self._start_cell.position

    @override
    def step(self, actions: list[Action]) -> TurnResult:
        if len(actions) == 0 or len(actions) > 5:
            raise ValueError(f"actions must have 1–5 entries, got {len(actions)}")

        self._fire_turn += 1
        active_fire = self._active_fire_cells()

        local_confusion = self._confusion_steps
        self._confusion_steps = 0

        result = TurnResult()

        for action in actions:
            if action == Action.WAIT:
                result.actions_executed += 1
                continue

            effective = action
            if local_confusion > 0:
                effective = _CONFUSION_FLIP[action]
                local_confusion -= 1

            next_cell: MazeCell | None = getattr(
                self._current_cell, _ACTION_TO_NEIGHBOR[effective]
            )
            if next_cell is None:
                result.wall_hits += 1
                result.actions_executed += 1
                continue

            self._current_cell = next_cell
            self._explored.add(self._current_cell.position)
            result.actions_executed += 1

            if self._current_cell.position in active_fire:
                result.is_dead = True
                self._deaths += 1
                self._current_cell = self._start_cell
                self._explored.add(self._start_cell.position)
                local_confusion = 0
                continue

            if self._current_cell.state == CellState.CONFUSION:
                result.is_confused = True
                self._confused_count += 1
                self._confusion_steps = 5
                local_confusion = 5

            dest = self._teleport_map.get(self._current_cell.position)
            if dest is not None:
                self._current_cell = dest
                self._explored.add(dest.position)
                result.teleported = True

            if self._current_cell.position == self._goal_pos:
                result.is_goal_reached = True
                self._goal_reached = True
                break

        result.current_position = self._current_cell.position
        self._turns += 1
        return result

    @override
    def get_episode_stats(self) -> dict:
        return {
            'turns_taken':    self._turns,
            'deaths':         self._deaths,
            'confused':       self._confused_count,
            'cells_explored': len(self._explored),
            'goal_reached':   self._goal_reached,
        }


# ── Maze generation ──────────────────────────────────────────────────────────

def generate_maze(
    seed: int,
    size: int = 64,
    num_fire_hazards: int = 8,
    num_teleport_pairs: int = 2,
    num_confusion: int = 5,
) -> list[list[MazeCell]]:
    """
    Generate a random solvable maze using iterative randomised DFS.

    Returns a list[list[MazeCell]] — the same type produced by
    load_image.load_image_into_graph().  Every cell is a node; a wall is
    a missing edge (neighbour attribute == None).

    Parameters
    ----------
    seed:               RNG seed for reproducibility.
    size:               Grid dimension; output is size×size cells.
    num_fire_hazards:   Number of rotating V-shaped fire hazards (HAZARD cells).
    num_teleport_pairs: Number of teleport pairs (PURPLE + GREEN TELEPORT).
    num_confusion:      Number of CONFUSION cells.
    """
    rng = np.random.default_rng(seed)

    # ── Phase 1: build graph with all walls up (no edges yet) ─────────────────
    graph: list[list[MazeCell]] = [
        [MazeCell() for _ in range(size)] for _ in range(size)
    ]
    for r in range(size):
        for c in range(size):
            graph[r][c].position = (c, r)  # (x, y) = (col, row)

    # ── Phase 2: DFS — carve passages by adding edges ─────────────────────────
    visited: set[tuple[int, int]] = {(0, 0)}
    stack:   list[tuple[int, int]] = [(0, 0)]

    while stack:
        r, c = stack[-1]
        neighbors = [
            (r + dr, c + dc, fwd, bwd)
            for dr, dc, fwd, bwd in _CELL_DIRS
            if 0 <= r + dr < size and 0 <= c + dc < size
            and (r + dr, c + dc) not in visited
        ]
        if neighbors:
            idx = int(rng.integers(len(neighbors)))
            nr, nc, fwd, bwd = neighbors[idx]
            setattr(graph[r][c],   fwd, graph[nr][nc])
            setattr(graph[nr][nc], bwd, graph[r][c])
            visited.add((nr, nc))
            stack.append((nr, nc))
        else:
            stack.pop()

    # ── Phase 3: place START and GOAL ─────────────────────────────────────────
    start_cell = graph[size - 1][int(rng.integers(0, size))]
    goal_cell  = graph[0][int(rng.integers(0, size))]
    start_cell.state = CellState.START
    goal_cell.state  = CellState.GOAL

    # ── Phase 4: find critical path to shield from fire hazards ───────────────
    # DFS produces a spanning tree — exactly one path between any two cells.
    # A fire hazard on that path makes the maze unsolvable, so we protect it.
    critical_path = _trace_path(start_cell, goal_cell)

    # ── Phase 5: collect candidate cells ──────────────────────────────────────
    candidates: list[MazeCell] = [
        graph[r][c]
        for r in range(size) for c in range(size)
        if graph[r][c].state == CellState.EMPTY
        and graph[r][c] not in critical_path
    ]
    rng.shuffle(candidates)
    ci = 0

    # ── Phase 6: fire hazard centers ──────────────────────────────────────────
    for _ in range(num_fire_hazards):
        if ci >= len(candidates):
            break
        candidates[ci].state = CellState.HAZARD
        ci += 1

    # ── Phase 7: teleport pairs ───────────────────────────────────────────────
    for _ in range(num_teleport_pairs):
        if ci + 1 >= len(candidates):
            break
        candidates[ci].state     = CellState.PURPLE_TELEPORT
        candidates[ci + 1].state = CellState.GREEN_TELEPORT
        ci += 2

    # ── Phase 8: confusion cells ──────────────────────────────────────────────
    for _ in range(num_confusion):
        if ci >= len(candidates):
            break
        candidates[ci].state = CellState.CONFUSION
        ci += 1

    # ── Phase 9: validate solvability ─────────────────────────────────────────
    assert _bfs_reachable(start_cell, goal_cell), \
        "Maze generator produced an unsolvable maze — this should not happen."

    return graph


def _bfs_reachable(start: MazeCell, goal: MazeCell) -> bool:
    """BFS through graph edges, treating HAZARD cells as impassable."""
    visited: set[MazeCell] = {start}
    q: deque[MazeCell] = deque([start])
    while q:
        cell = q.popleft()
        if cell is goal:
            return True
        for _, _, fwd, _ in _CELL_DIRS:
            nb: MazeCell | None = getattr(cell, fwd)
            if nb is not None and nb not in visited and nb.state != CellState.HAZARD:
                visited.add(nb)
                q.append(nb)
    return False


def _trace_path(start: MazeCell, goal: MazeCell) -> set[MazeCell]:
    """BFS path tracing; called before hazards are placed so all edges lead to EMPTY cells."""
    parent: dict[MazeCell, MazeCell | None] = {start: None}
    q: deque[MazeCell] = deque([start])
    while q:
        cell = q.popleft()
        if cell is goal:
            path: set[MazeCell] = set()
            cur: MazeCell | None = cell
            while cur is not None:
                path.add(cur)
                cur = parent[cur]
            return path
        for _, _, fwd, _ in _CELL_DIRS:
            nb: MazeCell | None = getattr(cell, fwd)
            if nb is not None and nb not in parent:
                parent[nb] = cell
                q.append(nb)
    return set()
