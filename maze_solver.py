from __future__ import annotations
from typing import override, NamedTuple
from dataclasses import dataclass
from enum import Enum
from load_image import CellState, MazeCell, load_image_into_graph
from solve_maze import get_start_and_goal_pos
from heapq import heappop, heappush, heapify

class Action(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4


class TurnResult:
    def __init__(self):
        self.wall_hits: int = 0
        self.current_position: tuple[int, int] = (0, 0)
        self.is_dead: bool = False
        self.is_confused: bool = False
        self.is_goal_reached: bool = False
        self.teleported: bool = False
        self.actions_executed: int = 0


class MazeEnvironment:
    def __init__(self, maze_id: str):
        """
        Initialize maze environment

        Args:
            maze_id: 'training' or 'testing'
        """
        pass

    def reset(self) -> tuple[int, int]:
        """
        Reset environment for new episode

        Returns:
            Starting position coordinates
        """
        pass

    def step(self, actions: list[Action]) -> TurnResult:
        """
        Execute a turn with given actions

        Args:
            actions: List of 1-5 Action objects

        Returns:
            TurnResult with feedback

        Raises:
            ValueError: If actions list empty or >5 actions
        """
        pass

    def get_episode_stats(self) -> dict:
        """
        Get statistics for current episode

        Returns:
            Dictionary containing:
            - turns_taken: int
            - deaths: int
            - confused: int
            - cells_explored: int
            - goal_reached: bool
        """
        pass


_MAZE_FILES = {
    'training': 'MAZE_1.png',
    'manual_hazards': 'MAZE_2.png',
    'testing':  'MAZE_0.png',
}

_ACTION_TO_NEIGHBOR = {
    Action.MOVE_UP: 'top_square',
    Action.MOVE_DOWN: 'bottom_square',
    Action.MOVE_LEFT:  'left_square',
    Action.MOVE_RIGHT: 'right_square',
}

_CONFUSION_OPPOSITE = {
    Action.MOVE_UP: Action.MOVE_DOWN,
    Action.MOVE_DOWN: Action.MOVE_UP,
    Action.MOVE_LEFT: Action.MOVE_RIGHT,
    Action.MOVE_RIGHT: Action.MOVE_LEFT,
    Action.WAIT: Action.WAIT,
}

_TELEPORT_STATES = {CellState.PURPLE_TELEPORT, CellState.GREEN_TELEPORT, CellState.YELLOW_TELEPORT}


def _find_fire_groups(graph: list) -> list[dict]:
    rows = len(graph)
    cols = len(graph[0]) if rows > 0 else 0
    hazard_set: set[tuple[int, int]] = set()
    for r in range(rows):
        for c in range(cols):
            if graph[r][c].state == CellState.HAZARD:
                hazard_set.add((c, r))  # store as (x, y)

    visited: set[tuple[int, int]] = set()
    groups: list[dict] = []
    for pos in hazard_set:
        if pos in visited:
            continue
        cluster: set[tuple[int, int]] = set()
        queue = [pos]
        while queue:
            cur = queue.pop()
            if cur in cluster:
                continue
            cluster.add(cur)
            cx, cy = cur
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nb = (cx + dx, cy + dy)
                    if nb in hazard_set and nb not in cluster:
                        queue.append(nb)
        visited |= cluster

        center = None
        for cell in cluster:
            cx, cy = cell
            nb_dirs = [(dx, dy)
                       for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                       if not (dx == 0 and dy == 0) and (cx + dx, cy + dy) in cluster]
            if len(nb_dirs) == 2:
                d1, d2 = nb_dirs
                if d1 != (-d2[0], -d2[1]):  # not collinear — this is the V center
                    center = cell
                    break

        if center is None:
            continue

        arms = [(ax - center[0], ay - center[1]) for ax, ay in cluster if (ax, ay) != center]
        groups.append({'center': center, 'arms': arms})

    return groups


class AStarMazeEnvironment(MazeEnvironment):
    def __init__(self, maze_id: str):
        self._graph = load_image_into_graph(_MAZE_FILES[maze_id])
        start_rc, goal_rc = get_start_and_goal_pos(self._graph)
        self._start_cell: MazeCell = self._graph[start_rc[0]][start_rc[1]]
        self._goal_pos: tuple[int, int] = (goal_rc[1], goal_rc[0])  # (x, y)
        AStarAgentMemoryCell.goal_pos = CellPosition(*self._goal_pos)
        self._current_cell: MazeCell = self._start_cell
        self._confusion_steps: int = 0
        self._turns: int = 0
        self._deaths: int = 0
        self._confused: int = 0
        self._explored: set[tuple[int, int]] = set()
        self._goal_reached: bool = False
        self._fire_groups: list[dict] = _find_fire_groups(self._graph)
        self._fire_turn: int = 0

    def _active_fire_cells(self) -> set[tuple[int, int]]:
        active: set[tuple[int, int]] = set()
        for g in self._fire_groups:
            cx, cy = g['center']
            active.add((cx, cy))
            for (dx, dy) in g['arms']:
                for _ in range(self._fire_turn % 4):
                    dx, dy = -dy, dx  # 90° CW in image coords
                active.add((cx + dx, cy + dy))
        return active

    def reset(self) -> tuple[int, int]:
        self._current_cell = self._start_cell
        self._confusion_steps = 0
        self._turns = 0
        self._deaths = 0
        self._confused = 0
        self._explored = {self._start_cell.position}
        self._goal_reached = False
        return self._start_cell.position

    def step(self, actions: list[Action]) -> TurnResult:
        if len(actions) == 0 or len(actions) > 5:
            raise ValueError(f"actions must have 1–5 entries, got {len(actions)}")

        self._confusion_steps = 0
        self._fire_turn += 1
        result = TurnResult()

        for action in actions:
            if action == Action.WAIT:
                result.actions_executed += 1
                continue

            if self._confusion_steps > 0:
                self._confusion_steps -= 1
                action = _CONFUSION_OPPOSITE[action]

            next_cell: MazeCell | None = getattr(self._current_cell, _ACTION_TO_NEIGHBOR[action])
            if next_cell is None:
                result.wall_hits += 1
                result.actions_executed += 1
                continue

            self._current_cell = next_cell
            self._explored.add(self._current_cell.position)
            result.actions_executed += 1

            if self._current_cell.position in self._active_fire_cells():
                result.is_dead = True
                self._deaths += 1
                self._current_cell = self._start_cell
                self._explored.add(self._start_cell.position)
                break

            if self._current_cell.state == CellState.CONFUSION:
                result.is_confused = True
                self._confused += 1
                self._confusion_steps = 5

            if self._current_cell.state in _TELEPORT_STATES:
                result.teleported = True

            if self._current_cell.position == self._goal_pos:
                result.is_goal_reached = True
                self._goal_reached = True
                print("goal reached!!! yay!!!!")
                break

        result.current_position = self._current_cell.position
        self._turns += 1
        return result

    def get_episode_stats(self) -> dict:
        return {
            'turns_taken': self._turns,
            'deaths': self._deaths,
            'confused': self._confused,
            'cells_explored': len(self._explored),
            'goal_reached': self._goal_reached,
        }


class Agent:
    """
    Base class for student implementations
    Students must implement this interface
    """

    def __init__(self):
        """
        Initialize agent with empty memory
        """
        self.memory = {}  # Students can structure as needed

    def plan_turn(self, last_result: TurnResult) -> list[Action]:
        """
        Plan next set of actions based on last turn result

        Args:
            last_result: Feedback from previous turn
                (None on first turn of episode)

        Returns:
            List of 1-5 actions to execute
        """
        raise NotImplementedError("Students must implement this method")

    def reset_episode(self):
        """
        Called at start of new episode
        Students can reset episode-specific state
        Memory can be retained for learning
        """
        pass


class CellPosition(NamedTuple):
    x: int
    y: int

@dataclass
class CellNeighbors:
    up: CellPosition | None = None
    right: CellPosition | None = None
    down: CellPosition | None = None
    left: CellPosition | None = None

class AStarAgentExpansionState(Enum):
    NOT_EXPANDING = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

class AStarAgentPlanningState(Enum):
    EXPANDING = 0
    TRAVERSING = 1
    START = 2
    RECOVERING = 3
    TELEPORT_RETURNING = 4

_EXPANSION_OFFSETS: dict[AStarAgentExpansionState, tuple[int, int]] = {
    AStarAgentExpansionState.UP: (0, -1),
    AStarAgentExpansionState.RIGHT: (1, 0),
    AStarAgentExpansionState.DOWN: (0, 1),
    AStarAgentExpansionState.LEFT: (-1, 0),
}

_EXPANSION_ORDER = [
    AStarAgentExpansionState.UP,
    AStarAgentExpansionState.RIGHT,
    AStarAgentExpansionState.DOWN,
    AStarAgentExpansionState.LEFT,
]

class AStarAgentMemoryCell:
    goal_pos: CellPosition = (0, 0)

    def __init__(self, state: CellState, pos: CellPosition, previous: AStarAgentMemoryCell | None):
        self.state = state
        self.pos = pos
        self.previous = previous
        self.neighbors = CellNeighbors()
        self.fully_expanded: bool = False
        self.cost: int = previous.cost + 1 if previous is not None else 0
        self.manhattan_distance = abs(AStarAgentMemoryCell.goal_pos.x - self.pos.x) + abs(AStarAgentMemoryCell.goal_pos.y - self.pos.y)
    
    def __repr__(self):
        return f"({self.pos.x}, {self.pos.y})"

    def __lt__(self, other: AStarAgentMemoryCell):
        return self.cost + self.manhattan_distance < other.cost + other.manhattan_distance

class AStarAgent(Agent):
    @override
    def __init__(self):
        self.memory: dict[CellPosition, AStarAgentMemoryCell] = {}
        self.plan_state: AStarAgentPlanningState = AStarAgentPlanningState.START
        self.expansion_state: AStarAgentExpansionState = AStarAgentExpansionState.NOT_EXPANDING
        self.current_path: list[CellPosition] = []
        self.open_queue: list[AStarAgentMemoryCell] = []
        self.visited: set[CellPosition] = set()
        self.expanding_cell: AStarAgentMemoryCell | None = None
        self.start_pos: CellPosition | None = None

        # fire tracking
        self.fire_map: dict[CellPosition, list[int]] = {}
        self.fire_counter: int = 0
        self.confusion_cells: set[CellPosition] = set()

        # recovery after fire death during expansion
        self._recovery_path: list[CellPosition] = []
        self._fire_check_direction: AStarAgentExpansionState | None = None
        self._fire_check_pending: bool = False  # True on the re-probe turn
        self._skip_fire_recheck: bool = False  # True when fire is impassable; skip re-probe on arrival

        # teleport return
        self._teleport_exit_order: list[Action] = [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT]
        self._teleport_exit_tried: int = 0
        self._teleport_last_exit: Action | None = None
        self._teleport_origin_dir: AStarAgentExpansionState | None = None
        self._teleport_stepping_back: bool = False  # True on step-back turn

        # traversal fire tracking
        self._last_turn_planned_cells: list[CellPosition] = []

    def _probe_cell_pos(self, direction: AStarAgentExpansionState) -> CellPosition:
        ex, ey = self.expanding_cell.pos
        dx, dy = _EXPANSION_OFFSETS[direction]
        return CellPosition(ex + dx, ey + dy)

    def _expand(self, last_result: TurnResult) -> list[Action]:
        # Return action reverses the previous probe; probe action moves into next direction.
        # When expanding_cell is a confusion cell, entering it (return move) confuses the agent
        # for subsequent actions in the turn, so the probe action must be pre-inverted.
        on_confusion = (self.expanding_cell is not None and
                        self.expanding_cell.state == CellState.CONFUSION)
        # Previous probe was open (agent is away from expanding_cell and will return through it)
        prev_was_open = last_result.wall_hits == 0

        if self.expansion_state == AStarAgentExpansionState.NOT_EXPANDING:
            self.expansion_state = AStarAgentExpansionState.UP
            # Start on expanding_cell (possibly confusion) — moving off it, not confused
            return [Action.MOVE_UP, Action.WAIT, Action.WAIT, Action.WAIT, Action.WAIT]

        elif self.expansion_state == AStarAgentExpansionState.UP:
            self.expansion_state = AStarAgentExpansionState.RIGHT
            probe = Action.MOVE_RIGHT
            if on_confusion and prev_was_open:
                probe = _CONFUSION_OPPOSITE[probe]
            actions = [Action.MOVE_DOWN, probe, Action.WAIT, Action.WAIT, Action.WAIT]
            if not prev_was_open:
                actions[0] = Action.WAIT
            return actions

        elif self.expansion_state == AStarAgentExpansionState.RIGHT:
            self.expansion_state = AStarAgentExpansionState.DOWN
            probe = Action.MOVE_DOWN
            if on_confusion and prev_was_open:
                probe = _CONFUSION_OPPOSITE[probe]
            actions = [Action.MOVE_LEFT, probe, Action.WAIT, Action.WAIT, Action.WAIT]
            if not prev_was_open:
                actions[0] = Action.WAIT
            return actions

        elif self.expansion_state == AStarAgentExpansionState.DOWN:
            self.expansion_state = AStarAgentExpansionState.LEFT
            probe = Action.MOVE_LEFT
            if on_confusion and prev_was_open:
                probe = _CONFUSION_OPPOSITE[probe]
            actions = [Action.MOVE_UP, probe, Action.WAIT, Action.WAIT, Action.WAIT]
            if not prev_was_open:
                actions[0] = Action.WAIT
            return actions

        else:  # LEFT
            self.expansion_state = AStarAgentExpansionState.NOT_EXPANDING
            actions = [Action.MOVE_RIGHT, Action.WAIT, Action.WAIT, Action.WAIT, Action.WAIT]
            if not prev_was_open:
                actions[0] = Action.WAIT
            return actions

    def _traverse(self, premade_actions: list[Action] | None = None) -> list[Action]:
        actions = premade_actions if premade_actions is not None else []
        self._last_turn_planned_cells = []
        confused_this_turn = False

        while len(actions) < 5 and len(self.current_path) > 1:
            current_x, current_y = self.current_path[0]
            next_x, next_y = self.current_path[1]
            next_pos = CellPosition(next_x, next_y)

            # Proactive fire check: stop if the cell is unsafe this turn OR next turn.
            # The +2 look-ahead prevents the agent from ending a turn on a cell that
            # will be on fire when the expansion return move steps back onto it.
            if next_pos in self.fire_map:
                safe_counters = set(range(4)) - set(self.fire_map[next_pos])
                if ((self.fire_counter + 1) % 4 not in safe_counters
                        or (self.fire_counter + 2) % 4 not in safe_counters):
                    break

            self.current_path.pop(0)
            self._last_turn_planned_cells.append(next_pos)

            # Use the stored neighbor relationship to determine the action direction.
            # Coordinate differences break for teleport edges where the destination
            # cell may be far away in coordinate space but only reachable by stepping
            # onto the teleporter (which is adjacent in the probe direction).
            current_cell = self.memory.get(CellPosition(current_x, current_y))
            if (current_cell is not None
                    and current_cell.neighbors.right is not None
                    and current_cell.neighbors.right.pos == next_pos):
                action = Action.MOVE_RIGHT
            elif (current_cell is not None
                    and current_cell.neighbors.left is not None
                    and current_cell.neighbors.left.pos == next_pos):
                action = Action.MOVE_LEFT
            elif (current_cell is not None
                    and current_cell.neighbors.down is not None
                    and current_cell.neighbors.down.pos == next_pos):
                action = Action.MOVE_DOWN
            elif (current_cell is not None
                    and current_cell.neighbors.up is not None
                    and current_cell.neighbors.up.pos == next_pos):
                action = Action.MOVE_UP
            elif next_x > current_x:
                action = Action.MOVE_RIGHT
            elif next_x < current_x:
                action = Action.MOVE_LEFT
            elif next_y > current_y:
                action = Action.MOVE_DOWN
            else:
                action = Action.MOVE_UP

            if confused_this_turn:
                action = _CONFUSION_OPPOSITE[action]

            actions.append(action)

            if next_pos in self.confusion_cells:
                confused_this_turn = True

        if len(self.current_path) == 1:
            destination_pos = self.current_path[0]
            self.expanding_cell = self.memory[destination_pos]
            self.visited.add(destination_pos)
            self.plan_state = AStarAgentPlanningState.EXPANDING
            self.current_path = []
            if self.expanding_cell.pos == AStarAgentMemoryCell.goal_pos:
                print("solution found!!")
                self.expanding_cell.manhattan_distance = -100000

        while len(actions) < 5:
            actions.append(Action.WAIT)

        return actions

    def _get_route_to_cell(self, start: AStarAgentMemoryCell, destination: AStarAgentMemoryCell) -> list[CellPosition]:
        journey_part_2: list[AStarAgentMemoryCell] = []
        curr: AStarAgentMemoryCell | None = destination
        while curr is not None:
            journey_part_2.append(curr)
            curr = curr.previous

        journey_part_2.reverse()

        journey_part_1: list[AStarAgentMemoryCell] = []
        curr = start
        while curr not in journey_part_2:
            journey_part_1.append(curr)
            curr = curr.previous

        path = journey_part_1 + journey_part_2[journey_part_2.index(curr):]
        return [cell.pos for cell in path]

    def _register_neighbor(self, new_cell: AStarAgentMemoryCell, direction: AStarAgentExpansionState):
        if direction == AStarAgentExpansionState.UP:
            new_cell.neighbors.down = self.expanding_cell
            self.expanding_cell.neighbors.up = new_cell
        elif direction == AStarAgentExpansionState.RIGHT:
            new_cell.neighbors.left = self.expanding_cell
            self.expanding_cell.neighbors.right = new_cell
        elif direction == AStarAgentExpansionState.DOWN:
            new_cell.neighbors.up = self.expanding_cell
            self.expanding_cell.neighbors.down = new_cell
        elif direction == AStarAgentExpansionState.LEFT:
            new_cell.neighbors.right = self.expanding_cell
            self.expanding_cell.neighbors.left = new_cell

    def _pop_and_traverse(self, premade_actions: list[Action] | None = None) -> list[Action]:
        def _go_to_goal():
            goal_cell = self.memory.get(AStarAgentMemoryCell.goal_pos)
            if (goal_cell is not None and self.expanding_cell is not None
                    and self.expanding_cell.pos != AStarAgentMemoryCell.goal_pos):
                self.current_path = self._get_route_to_cell(self.expanding_cell, goal_cell)
                self.plan_state = AStarAgentPlanningState.TRAVERSING
                return self._traverse(premade_actions)
            return (premade_actions or []) + [Action.WAIT] * (5 - len(premade_actions or []))

        if not self.open_queue:
            return _go_to_goal()
        cell_to_traverse_to = heappop(self.open_queue)
        while cell_to_traverse_to.fully_expanded and self.open_queue:
            cell_to_traverse_to = heappop(self.open_queue)
        if cell_to_traverse_to.fully_expanded:
            return _go_to_goal()
        self.current_path = self._get_route_to_cell(self.expanding_cell, cell_to_traverse_to)
        self.plan_state = AStarAgentPlanningState.TRAVERSING
        return self._traverse(premade_actions)

    def _traverse_recovery_path(self) -> list[Action]:
        actions: list[Action] = []
        self._last_turn_planned_cells = []
        while len(actions) < 5 and len(self._recovery_path) > 1:
            cx, cy = self._recovery_path[0]
            nx, ny = self._recovery_path[1]
            next_pos = CellPosition(nx, ny)
            if next_pos in self.fire_map:
                safe_counters = set(range(4)) - set(self.fire_map[next_pos])
                if ((self.fire_counter + 1) % 4 not in safe_counters
                        or (self.fire_counter + 2) % 4 not in safe_counters):
                    break
            self._recovery_path.pop(0)
            self._last_turn_planned_cells.append(next_pos)
            if nx > cx:
                actions.append(Action.MOVE_RIGHT)
            elif nx < cx:
                actions.append(Action.MOVE_LEFT)
            elif ny > cy:
                actions.append(Action.MOVE_DOWN)
            else:
                actions.append(Action.MOVE_UP)
        while len(actions) < 5:
            actions.append(Action.WAIT)
        return actions

    @override
    def plan_turn(self, last_result: TurnResult):
        # Increment fire counter at end of each turn; we read it here for THIS turn's planning
        pos = CellPosition(*last_result.current_position)

        # ── START ─────────────────────────────────────────────────────────────
        if self.plan_state == AStarAgentPlanningState.START:
            self.plan_state = AStarAgentPlanningState.EXPANDING
            self.start_pos = pos
            result = [Action.WAIT for _ in range(5)]
            self.fire_counter = (self.fire_counter + 1) % 4
            return result

        # ── RECOVERING (fire killed agent during expansion; navigate back) ─────
        if self.plan_state == AStarAgentPlanningState.RECOVERING:
            _dir_to_action = {
                AStarAgentExpansionState.UP: Action.MOVE_UP,
                AStarAgentExpansionState.RIGHT: Action.MOVE_RIGHT,
                AStarAgentExpansionState.DOWN: Action.MOVE_DOWN,
                AStarAgentExpansionState.LEFT: Action.MOVE_LEFT,
            }

            if last_result.is_dead and not self._fire_check_pending:
                # Died during recovery traversal — record the fire cell then rebuild path.
                # The proactive check in _traverse_recovery_path will then wait for a safe
                # counter before attempting that cell again.
                if self._last_turn_planned_cells:
                    fire_pos = self._last_turn_planned_cells[-1]
                    self.fire_map.setdefault(fire_pos, []).append(self.fire_counter)
                start_cell = self.memory[self.start_pos]
                self._recovery_path = self._get_route_to_cell(start_cell, self.expanding_cell)
                result = self._traverse_recovery_path()
                self.fire_counter = (self.fire_counter + 1) % 4
                return result

            _dir_return_action = {
                AStarAgentExpansionState.UP: Action.MOVE_DOWN,
                AStarAgentExpansionState.RIGHT: Action.MOVE_LEFT,
                AStarAgentExpansionState.DOWN: Action.MOVE_UP,
                AStarAgentExpansionState.LEFT: Action.MOVE_RIGHT,
            }

            if self._fire_check_pending:
                # Result of the re-probe: agent was at expanding_cell, probed fire direction
                fire_pos = self._probe_cell_pos(self._fire_check_direction)
                completed_dir = self._fire_check_direction
                self._fire_check_pending = False

                if last_result.is_dead:
                    # Fire still there — record counter; agent is at start_pos
                    self.fire_map.setdefault(fire_pos, []).append(self.fire_counter)
                    if set(self.fire_map[fire_pos]) == {0, 1, 2, 3}:
                        # Impassable — navigate back to expanding_cell then skip re-probe
                        self._skip_fire_recheck = True
                    start_cell = self.memory[self.start_pos]
                    self._recovery_path = self._get_route_to_cell(start_cell, self.expanding_cell)
                    result = self._traverse_recovery_path()
                    self.fire_counter = (self.fire_counter + 1) % 4
                    return result
                else:
                    # Passable — register the cell if agent actually moved (no wall)
                    agent_moved = last_result.wall_hits == 0
                    if agent_moved and fire_pos not in self.memory:
                        new_cell = AStarAgentMemoryCell(CellState.EMPTY, fire_pos, self.expanding_cell)
                        self.memory[fire_pos] = new_cell
                        self._register_neighbor(new_cell, completed_dir)
                        heappush(self.open_queue, new_cell)
                    idx = _EXPANSION_ORDER.index(completed_dir)
                    self._fire_check_direction = None
                    if idx + 1 >= len(_EXPANSION_ORDER):
                        self.expanding_cell.fully_expanded = True
                        self.plan_state = AStarAgentPlanningState.TRAVERSING
                        self.expansion_state = AStarAgentExpansionState.NOT_EXPANDING
                        if agent_moved:
                            return_action = _dir_return_action[completed_dir]
                            result = self._pop_and_traverse([return_action])
                        else:
                            result = self._pop_and_traverse()
                    else:
                        next_dir = _EXPANSION_ORDER[idx + 1]
                        self.expansion_state = next_dir
                        self.plan_state = AStarAgentPlanningState.EXPANDING
                        if agent_moved:
                            return_action = _dir_return_action[completed_dir]
                            result = [return_action, _dir_to_action[next_dir]] + [Action.WAIT] * 3
                        else:
                            result = [_dir_to_action[next_dir]] + [Action.WAIT] * 4
                    self.fire_counter = (self.fire_counter + 1) % 4
                    return result

            if len(self._recovery_path) > 1:
                result = self._traverse_recovery_path()
                self.fire_counter = (self.fire_counter + 1) % 4
                return result

            # Arrived at expanding_cell
            completed_dir = self._fire_check_direction
            if self._skip_fire_recheck:
                # Fire was impassable — skip re-probe, resume expansion from next direction
                self._skip_fire_recheck = False
                self._fire_check_direction = None
                idx = _EXPANSION_ORDER.index(completed_dir)
                if idx + 1 >= len(_EXPANSION_ORDER):
                    self.expanding_cell.fully_expanded = True
                    self.plan_state = AStarAgentPlanningState.TRAVERSING
                    self.expansion_state = AStarAgentExpansionState.NOT_EXPANDING
                    result = self._pop_and_traverse()
                else:
                    next_dir = _EXPANSION_ORDER[idx + 1]
                    self.expansion_state = next_dir
                    self.plan_state = AStarAgentPlanningState.EXPANDING
                    result = [_dir_to_action[next_dir]] + [Action.WAIT] * 4
                self.fire_counter = (self.fire_counter + 1) % 4
                return result

            # Re-probe the fire direction, but only when it's known-safe.
            # Without this check the agent walks into a cell it already knows
            # has fire whenever the cycle brings the dangerous counter back
            # around before the re-probe fires.
            # The +2 check prevents dying on the return move after a successful
            # probe, but is relaxed when only 1 safe counter remains so the
            # agent can sample the final counter and mark center cells (always
            # burning) as impassable instead of looping forever.
            fire_pos = self._probe_cell_pos(self._fire_check_direction)
            if fire_pos in self.fire_map:
                safe_counters = set(range(4)) - set(self.fire_map[fire_pos])
                if not safe_counters:
                    # Cell is fully mapped as impassable. Set the skip flag and
                    # wait one turn so the _skip_fire_recheck handler above fires
                    # next turn instead of this gate looping forever.
                    self._skip_fire_recheck = True
                    result = [Action.WAIT] * 5
                    self.fire_counter = (self.fire_counter + 1) % 4
                    return result
                # A safe probe requires two *consecutive* safe ticks: one to
                # step in, one to step back.  If no such pair exists (e.g.
                # safe_counters = {0,2} from overlapping fire arms), the cell
                # can never be probed safely — skip it rather than waiting forever.
                has_consecutive_safe = any(
                    (c + 1) % 4 in safe_counters for c in safe_counters
                )
                if not has_consecutive_safe:
                    self._skip_fire_recheck = True
                    result = [Action.WAIT] * 5
                    self.fire_counter = (self.fire_counter + 1) % 4
                    return result
                next_unsafe = (self.fire_counter + 1) % 4 not in safe_counters
                next_next_unsafe = (self.fire_counter + 2) % 4 not in safe_counters
                if next_unsafe or (next_next_unsafe and len(safe_counters) > 1):
                    result = [Action.WAIT] * 5
                    self.fire_counter = (self.fire_counter + 1) % 4
                    return result
            self._fire_check_pending = True
            result = [_dir_to_action[self._fire_check_direction]] + [Action.WAIT] * 4
            self.fire_counter = (self.fire_counter + 1) % 4
            return result

        # ── TELEPORT_RETURNING ────────────────────────────────────────────────
        if self.plan_state == AStarAgentPlanningState.TELEPORT_RETURNING:
            if last_result.teleported and self._teleport_stepping_back:
                # Accidental teleport: Turn A set _teleport_stepping_back=True then the
                # exit action itself landed on another teleporter before we could step
                # back.  The agent is at an unrelated destination — treat as blocked and
                # try the next exit direction on the next Turn A.
                self._teleport_exit_tried += 1
                self._teleport_stepping_back = False
                # fall through to Turn A below

            elif last_result.teleported:
                # Turn B's step-back reached B which teleported us to the original
                # teleporter (A). _teleport_stepping_back was reset to False by Turn B
                # before the step fired, so !_teleport_stepping_back signals success.
                # Return to expanding_cell and resume expansion.
                origin_dir = self._teleport_origin_dir
                ret_action = {
                    AStarAgentExpansionState.UP: Action.MOVE_DOWN,
                    AStarAgentExpansionState.RIGHT: Action.MOVE_LEFT,
                    AStarAgentExpansionState.DOWN: Action.MOVE_UP,
                    AStarAgentExpansionState.LEFT: Action.MOVE_RIGHT,
                }[origin_dir]
                idx = _EXPANSION_ORDER.index(origin_dir)
                self._teleport_in_progress = False
                self._teleport_origin_dir = None
                self._teleport_exit_tried = 0
                self._teleport_last_exit = None
                self._teleport_stepping_back = False
                self.plan_state = AStarAgentPlanningState.EXPANDING
                _dir_to_action = {
                    AStarAgentExpansionState.UP: Action.MOVE_UP,
                    AStarAgentExpansionState.RIGHT: Action.MOVE_RIGHT,
                    AStarAgentExpansionState.DOWN: Action.MOVE_DOWN,
                    AStarAgentExpansionState.LEFT: Action.MOVE_LEFT,
                }
                if idx + 1 >= len(_EXPANSION_ORDER):
                    # Was the last direction — finish expansion and pick next cell.
                    # Must call _pop_and_traverse (not just set TRAVERSING) so the
                    # open queue is consulted; without it current_path stays empty
                    # and the agent waits forever.
                    self.expanding_cell.fully_expanded = True
                    self.expansion_state = AStarAgentExpansionState.NOT_EXPANDING
                    result = self._pop_and_traverse([ret_action])
                else:
                    next_dir = _EXPANSION_ORDER[idx + 1]
                    self.expansion_state = next_dir
                    result = [ret_action, _dir_to_action[next_dir]] + [Action.WAIT] * 3
                self.fire_counter = (self.fire_counter + 1) % 4
                return result

            if self._teleport_stepping_back:
                # Turn B: we previously stepped off — now step back onto teleport cell
                if last_result.wall_hits > 0:
                    # Step-off was blocked; try next direction
                    self._teleport_exit_tried += 1
                    self._teleport_stepping_back = False
                elif last_result.is_confused:
                    # Stepped onto a confusion cell; return action must be sent as-is
                    # (confusion will invert it to the correct opposite)
                    self.confusion_cells.add(CellPosition(*last_result.current_position))
                    step_back = self._teleport_last_exit  # same dir; confusion inverts it
                    result = [step_back] + [Action.WAIT] * 4
                    self._teleport_stepping_back = False
                    self.fire_counter = (self.fire_counter + 1) % 4
                    return result
                else:
                    step_back = _CONFUSION_OPPOSITE[self._teleport_last_exit]
                    result = [step_back] + [Action.WAIT] * 4
                    self._teleport_stepping_back = False
                    self.fire_counter = (self.fire_counter + 1) % 4
                    return result

            # Turn A: try to step off teleport destination
            if self._teleport_exit_tried >= len(self._teleport_exit_order):
                # All directions blocked — stuck; just wait
                result = [Action.WAIT] * 5
                self.fire_counter = (self.fire_counter + 1) % 4
                return result
            exit_action = self._teleport_exit_order[self._teleport_exit_tried]
            self._teleport_last_exit = exit_action
            self._teleport_stepping_back = True
            result = [exit_action] + [Action.WAIT] * 4
            self.fire_counter = (self.fire_counter + 1) % 4
            return result

        # ── EXPANDING ─────────────────────────────────────────────────────────
        if self.plan_state == AStarAgentPlanningState.EXPANDING:
            # Detect traversal death: _traverse eagerly set plan_state=EXPANDING, but agent died
            if (self.expansion_state == AStarAgentExpansionState.NOT_EXPANDING
                    and last_result.is_dead
                    and self.expanding_cell is not None
                    and pos != self.expanding_cell.pos):
                if self._last_turn_planned_cells:
                    fire_pos = self._last_turn_planned_cells[-1]
                    self.fire_map.setdefault(fire_pos, []).append(self.fire_counter)
                start_cell = self.memory[self.start_pos]
                self.current_path = self._get_route_to_cell(start_cell, self.expanding_cell)
                self.plan_state = AStarAgentPlanningState.TRAVERSING
                result = self._traverse()
                self.fire_counter = (self.fire_counter + 1) % 4
                return result

            if self.expanding_cell is None:
                start_cell = self.memory.get(pos) or AStarAgentMemoryCell(CellState.EMPTY, pos, None)
                self.memory[pos] = start_cell
                self.visited.add(pos)
                self.expanding_cell = start_cell

            if self.expansion_state != AStarAgentExpansionState.NOT_EXPANDING:
                if last_result.is_dead:
                    # Fire found at probe cell
                    fire_pos = self._probe_cell_pos(self.expansion_state)
                    self.fire_map.setdefault(fire_pos, []).append(self.fire_counter)
                    self._fire_check_direction = self.expansion_state
                    start_cell = self.memory[self.start_pos]
                    self._recovery_path = self._get_route_to_cell(start_cell, self.expanding_cell)
                    self.plan_state = AStarAgentPlanningState.RECOVERING
                    result = [Action.WAIT] * 5
                    self.fire_counter = (self.fire_counter + 1) % 4
                    return result

                elif last_result.teleported:
                    # Discovered a teleporter — use actual position for new cell
                    teleport_dest = CellPosition(*last_result.current_position)
                    if teleport_dest not in self.memory:
                        new_cell = AStarAgentMemoryCell(CellState.EMPTY, teleport_dest, self.expanding_cell)
                        self.memory[teleport_dest] = new_cell
                        self._register_neighbor(new_cell, self.expansion_state)
                        heappush(self.open_queue, new_cell)
                    self._teleport_origin_dir = self.expansion_state
                    self._teleport_exit_tried = 0
                    self._teleport_stepping_back = False
                    self.plan_state = AStarAgentPlanningState.TELEPORT_RETURNING
                    result = [Action.WAIT] * 5
                    self.fire_counter = (self.fire_counter + 1) % 4
                    return result

                elif last_result.wall_hits == 0 and pos not in self.memory:
                    # Normal discovery
                    cell_state = CellState.CONFUSION if last_result.is_confused else CellState.EMPTY
                    new_cell = AStarAgentMemoryCell(cell_state, pos, self.expanding_cell)
                    self.memory[pos] = new_cell
                    self._register_neighbor(new_cell, self.expansion_state)
                    if last_result.is_confused:
                        self.confusion_cells.add(pos)
                    heappush(self.open_queue, new_cell)

            # Pre-check: if the next probe cell is already in fire_map and will
            # be on fire next turn, don't probe it — divert to RECOVERING so
            # the re-probe gate handles timing.  This prevents the agent from
            # repeatedly walking into a known-fire cell during expansion.
            _next_probe_dir = {
                AStarAgentExpansionState.NOT_EXPANDING: AStarAgentExpansionState.UP,
                AStarAgentExpansionState.UP:            AStarAgentExpansionState.RIGHT,
                AStarAgentExpansionState.RIGHT:         AStarAgentExpansionState.DOWN,
                AStarAgentExpansionState.DOWN:          AStarAgentExpansionState.LEFT,
            }.get(self.expansion_state)
            if _next_probe_dir is not None:
                _next_probe_pos = self._probe_cell_pos(_next_probe_dir)
                if _next_probe_pos in self.fire_map:
                    _safe = set(range(4)) - set(self.fire_map[_next_probe_pos])
                    if (self.fire_counter + 1) % 4 not in _safe:
                        self._fire_check_direction = _next_probe_dir
                        if pos != self.expanding_cell.pos:
                            self._recovery_path = [pos, self.expanding_cell.pos]
                        else:
                            self._recovery_path = [self.expanding_cell.pos]
                        # If the cell is fully mapped as impassable (no safe counter),
                        # flag it so the RECOVERING skip-handler fires on arrival
                        # instead of the re-probe gate looping forever.
                        if not _safe:
                            self._skip_fire_recheck = True
                        self.plan_state = AStarAgentPlanningState.RECOVERING
                        result = [Action.WAIT] * 5
                        self.fire_counter = (self.fire_counter + 1) % 4
                        return result

            next_actions = self._expand(last_result)
            if self.expansion_state == AStarAgentExpansionState.NOT_EXPANDING:
                self.plan_state = AStarAgentPlanningState.TRAVERSING
                self.expanding_cell.fully_expanded = True
                result = self._pop_and_traverse(next_actions)
                self.fire_counter = (self.fire_counter + 1) % 4
                return result

            result = next_actions
            self.fire_counter = (self.fire_counter + 1) % 4
            return result

        # ── TRAVERSING ────────────────────────────────────────────────────────
        if self.plan_state == AStarAgentPlanningState.TRAVERSING:
            if last_result.is_dead:
                # Fire tracking only when traversal cells were planned this turn.
                # When death happens on the expansion-return move, _last_turn_planned_cells
                # is empty (the while-loop in _traverse never ran), so we skip recording
                # and still recalculate — the critical part that was previously gated on
                # a non-empty list.
                if self._last_turn_planned_cells:
                    fire_pos = self._last_turn_planned_cells[-1]
                    self.fire_map.setdefault(fire_pos, []).append(self.fire_counter)
                start_cell = self.memory[self.start_pos]
                dest = self.memory.get(self.current_path[-1]) if self.current_path else None
                if dest is None and self.expanding_cell:
                    dest = self.expanding_cell
                if dest:
                    self.current_path = self._get_route_to_cell(start_cell, dest)
            result = self._traverse()
            self.fire_counter = (self.fire_counter + 1) % 4
            return result

    @override
    def reset_episode(self):
        if self.expanding_cell is not None:
            heappush(self.open_queue, self.expanding_cell)

        self.expansion_state = AStarAgentExpansionState.NOT_EXPANDING
        self.current_path = []
        self.visited = {pos for pos, cell in self.memory.items() if cell.fully_expanded}
        self.open_queue = [cell for cell in self.memory.values() if not cell.fully_expanded]
        self.expanding_cell = None
        heapify(self.open_queue)

        # Reset episode-local state; fire_map, confusion_cells, fire_counter persist
        self._recovery_path = []
        self._fire_check_direction = None
        self._fire_check_pending = False
        self._skip_fire_recheck = False
        self._teleport_exit_tried = 0
        self._teleport_last_exit = None
        self._teleport_origin_dir = None
        self._teleport_stepping_back = False
        self._last_turn_planned_cells = []

        self.plan_state = AStarAgentPlanningState.START

class ManualMovementAgent(Agent):
    @override
    def __init__(self):
        self.memory: set[CellPosition] = {}
        self.next_action: Action

    @override
    def plan_turn(self, _):
        ans = self.next_action
        self.next_action = Action.WAIT
        return [ans]
    
    def set_action(self, action: Action):
        self.next_action = action
