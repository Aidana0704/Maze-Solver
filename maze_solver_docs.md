# Maze Solver Documentation

## Overview

`maze_solver.py` implements a maze-solving agent that uses an **online A\* search** strategy: it explores the maze step-by-step, building an internal map as it goes, and navigates toward the goal once enough of the graph is known. The environment supports hazards (rotating fire), confusion cells, and teleporters, all of which the agent must learn to handle during live play.

---

## Enums

### `Action`

The five actions an agent can submit in a single turn slot.

| Value | Meaning |
|---|---|
| `MOVE_UP` | Step one cell upward |
| `MOVE_DOWN` | Step one cell downward |
| `MOVE_LEFT` | Step one cell to the left |
| `MOVE_RIGHT` | Step one cell to the right |
| `WAIT` | Do nothing for this slot |

Each turn the agent submits a list of 1–5 `Action` values. The environment executes them in order within the same turn.

---

### `AStarAgentExpansionState`

Tracks which neighbor direction the agent is currently probing during cell expansion.

| Value | Meaning |
|---|---|
| `NOT_EXPANDING` | No probe in progress; expansion is idle or complete |
| `UP` | Currently probing the cell above |
| `RIGHT` | Currently probing the cell to the right |
| `DOWN` | Currently probing the cell below |
| `LEFT` | Currently probing the cell to the left |

Expansion always proceeds in the order UP → RIGHT → DOWN → LEFT. After LEFT the state returns to `NOT_EXPANDING` and the cell is marked fully expanded.

---

### `AStarAgentPlanningState`

The top-level state machine that controls `AStarAgent.plan_turn`. Each state represents a distinct behavioral mode.

| Value | Meaning |
|---|---|
| `START` | First turn of an episode — initialization only |
| `EXPANDING` | Probing the four neighbors of the current frontier cell |
| `TRAVERSING` | Following a planned path to the next cell to expand (or to the goal) |
| `RECOVERING` | Navigating back to a cell after fire killed the agent during expansion |
| `TELEPORT_RETURNING` | Escaping an unexpected teleport destination to resume expansion |

---

## Data Classes

### `TurnResult`

Feedback returned by `MazeEnvironment.step` after one turn executes.

| Field | Type | Description |
|---|---|---|
| `wall_hits` | `int` | Number of actions that tried to move into a wall |
| `current_position` | `tuple[int, int]` | Agent's `(x, y)` position at end of turn |
| `is_dead` | `bool` | Agent was killed by fire this turn and respawned at start |
| `is_confused` | `bool` | Agent stepped on a confusion cell this turn |
| `is_goal_reached` | `bool` | Agent reached the goal cell this turn |
| `teleported` | `bool` | Agent stepped onto a teleport cell and was moved |
| `actions_executed` | `int` | How many action slots were processed before the turn ended |

---

### `CellPosition`

A `NamedTuple` with fields `x: int` and `y: int`. Used as dictionary keys throughout the agent's memory. Positions are in `(x, y)` / `(col, row)` space.

---

### `CellNeighbors`

A `dataclass` that stores up to four `CellPosition` references (or `None`) for the cardinal neighbors of a memory cell. Used to determine the correct move direction when traversing, including across teleport edges where coordinate arithmetic would give the wrong answer.

| Field | Meaning |
|---|---|
| `up` | Position of the cell directly above |
| `right` | Position of the cell directly to the right |
| `down` | Position of the cell directly below |
| `left` | Position of the cell directly to the left |

---

### `AStarAgentMemoryCell`

One node in the agent's internal graph. Built lazily as the agent discovers new cells.

| Field | Type | Description |
|---|---|---|
| `state` | `CellState` | The type of cell (empty, confusion, teleport, etc.) |
| `pos` | `CellPosition` | Grid coordinates of this cell |
| `previous` | `AStarAgentMemoryCell \| None` | Parent pointer used to reconstruct paths |
| `neighbors` | `CellNeighbors` | Known adjacent cells |
| `fully_expanded` | `bool` | All four directions have been probed; cell is no longer on the open queue |
| `cost` | `int` | Depth from start (number of steps from origin, via parent chain) |
| `manhattan_distance` | `int` | Heuristic: Manhattan distance to goal |
| `goal_pos` | `CellPosition` *(class variable)* | Shared goal position used to compute heuristics for all cells |

`__lt__` compares by `cost + manhattan_distance`, making cells naturally sortable for a min-heap A\* open queue.

---

## Environment Classes

### `MazeEnvironment` (abstract base)

Defines the interface all environment implementations must provide.

#### `__init__(self, maze_id: str)`
Accepts `'training'` or `'testing'` to select a maze image file.

#### `reset() -> tuple[int, int]`
Resets the environment for a new episode. Returns the starting position `(x, y)`.

#### `step(actions: list[Action]) -> TurnResult`
Executes one turn. `actions` must have 1–5 entries; raises `ValueError` otherwise. Returns a `TurnResult` describing what happened.

#### `get_episode_stats() -> dict`
Returns a summary of the current episode:
- `turns_taken` — total turns elapsed
- `deaths` — how many times the agent was killed by fire
- `confused` — how many times a confusion cell was stepped on
- `cells_explored` — unique cells visited
- `goal_reached` — whether the goal was ever reached

---

### `AStarMazeEnvironment(MazeEnvironment)`

The full simulation. Loads the maze from a PNG image, parses hazard (fire) groups, and processes turn actions one at a time.

#### Fields

| Field | Description |
|---|---|
| `_graph` | 2-D grid of `MazeCell` objects loaded from the image |
| `_start_cell` | The starting `MazeCell` |
| `_goal_pos` | Goal position as `(x, y)` |
| `_current_cell` | The cell the agent currently occupies |
| `_confusion_steps` | Remaining turns during which movement is inverted |
| `_turns` | Turn counter for the current episode |
| `_deaths` | Death counter |
| `_confused` | Confusion event counter |
| `_explored` | Set of visited cell positions |
| `_goal_reached` | Whether the goal has been reached |
| `_fire_groups` | List of rotating fire cluster descriptors (center + arm offsets) |
| `_fire_turn` | Monotonically increasing counter that drives fire rotation |

#### `_active_fire_cells() -> set[tuple[int, int]]`

Computes which cells are on fire this turn. Each fire group has a center cell (always burning) and two arms. Every turn, the arms rotate 90° clockwise (in image coordinates), cycling through four positions. A cell is active fire if it is either the center or an arm endpoint at the current `_fire_turn % 4`.

#### `step(actions) -> TurnResult`

For each action in the list:
1. `WAIT` — counts as executed, does nothing else.
2. If `_confusion_steps > 0`, the action is inverted to its opposite before being applied.
3. The target neighbor cell is looked up. If it is `None` (wall), `wall_hits` is incremented.
4. If the move succeeds, the agent moves to the neighbor.
5. If the new cell is an active fire cell, the agent dies and is teleported back to start. Processing stops.
6. If the new cell is a confusion cell, `_confusion_steps` is set to 5 for the next turn.
7. If the new cell is a teleport cell, `teleported` is flagged (the graph node for a teleport cell stores the destination as its "neighbor").
8. If the new cell is the goal, `is_goal_reached` is set and processing stops.

---

## Agent Classes

### `Agent` (abstract base)

Defines the interface student implementations must follow.

| Field | Description |
|---|---|
| `memory` | A dictionary (or any structure) for the agent to store what it has learned |

#### `plan_turn(last_result: TurnResult) -> list[Action]`
Called once per turn. `last_result` is `None` on the very first turn. Must return a list of 1–5 `Action` values.

#### `reset_episode()`
Called at the start of each new episode. The agent may choose to retain cross-episode memory (e.g., a map) or reset it entirely.

---

### `AStarAgent(Agent)`

The main implementation. Uses online A\* search to explore and navigate the maze. Maintains a persistent map across episodes.

#### Fields

| Field | Description |
|---|---|
| `memory` | `dict[CellPosition, AStarAgentMemoryCell]` — the discovered graph |
| `plan_state` | Current `AStarAgentPlanningState` |
| `expansion_state` | Current `AStarAgentExpansionState` (which direction is being probed) |
| `current_path` | Ordered list of `CellPosition` waypoints for the active traversal |
| `open_queue` | Min-heap of unexpanded `AStarAgentMemoryCell` objects (A\* frontier) |
| `visited` | Set of positions whose cells have been fully expanded |
| `expanding_cell` | The `AStarAgentMemoryCell` currently being probed |
| `start_pos` | Position of the start cell, set on the first turn |
| `fire_map` | `dict[CellPosition, list[int]]` — records which `fire_counter % 4` values a cell was on fire |
| `fire_counter` | Local mirror of the environment's fire phase, incremented every turn |
| `confusion_cells` | Set of positions known to be confusion cells |
| `_recovery_path` | Waypoints for navigating back to `expanding_cell` after a death |
| `_fire_check_direction` | The expansion direction whose probe cell triggered fire recovery |
| `_fire_check_pending` | `True` on the turn when the agent re-probes after arriving back at `expanding_cell` |
| `_skip_fire_recheck` | `True` when a fire cell is fully mapped as impassable; skips the re-probe step |
| `_teleport_exit_order` | Directions to try when escaping an unexpected teleport destination |
| `_teleport_exit_tried` | How many exit directions have already been attempted |
| `_teleport_last_exit` | The exit action used on the previous Turn A |
| `_teleport_origin_dir` | The expansion direction that caused the teleport |
| `_teleport_stepping_back` | `True` on Turn B (stepping back toward the teleport pad) |
| `_last_turn_planned_cells` | Cells the agent intended to visit this traversal turn, used to identify which cell caused a death |

---

## Planning State Machine

`plan_turn` is a large conditional block structured as a state machine. Here is how each state behaves and how `last_result` drives transitions.

---

### `START`

**Entered:** First call to `plan_turn` per episode.

**Behavior:** Records the starting position, initializes `start_pos`, and returns five `WAIT` actions.

**Transition:** Always exits to `EXPANDING` immediately.

---

### `EXPANDING`

**Entered from:** `START` (initially), `TRAVERSING` (when path is exhausted and destination is reached), `RECOVERING` (after successfully re-probing a fire cell), `TELEPORT_RETURNING` (after returning from a teleport detour).

**Behavior:** Systematically probes each of the four neighbors of `expanding_cell` in UP → RIGHT → DOWN → LEFT order. Each probe is a two-action sequence: move into the neighbor, then move back. Because both actions happen in the same turn, each probe takes exactly one turn. The internal `_expand` helper manages the action sequencing.

**Effect of `last_result` on this state:**

| Condition | Action |
|---|---|
| `last_result.is_dead` and `expansion_state != NOT_EXPANDING` | Fire discovered at probe cell. Records the fire phase in `fire_map`, stores the direction in `_fire_check_direction`, builds a recovery path, transitions to `RECOVERING`. |
| `last_result.teleported` | Probe entered a teleport and the agent was moved. Records the teleport destination in memory (marked `fully_expanded` so it is never re-queued), stores the probe direction in `_teleport_origin_dir`, transitions to `TELEPORT_RETURNING`. |
| `last_result.wall_hits == 0` and position not yet in memory | Normal discovery. Creates an `AStarAgentMemoryCell` for the new cell, wires its neighbor pointers, and pushes it onto the A\* open queue. If the cell is a confusion cell, adds it to `confusion_cells`. |
| `expansion_state == NOT_EXPANDING` after all four probes | Marks `expanding_cell.fully_expanded = True`, then calls `_pop_and_traverse` to pick the next frontier cell and begin traversal. Transitions to `TRAVERSING`. |

**Pre-checks before probing:**

- If the next probe cell is already in `fire_map` and will be on fire next turn, the agent preemptively enters `RECOVERING` instead of probing.
- If `expanding_cell` is a teleport pad, expansion is skipped entirely (probing from a teleport pad is impossible because return steps rewire to the paired pad). The cell is immediately marked fully expanded and the agent moves to `TRAVERSING`.

---

### `TRAVERSING`

**Entered from:** `EXPANDING` (when `expansion_state` returns to `NOT_EXPANDING` and `_pop_and_traverse` is called), `RECOVERING` (after successful fire re-probe and expansion resumes).

**Behavior:** Follows `current_path` step by step. Up to 5 steps can be batched into a single turn. `_traverse` pops waypoints from `current_path`, determines the move direction for each step (using stored neighbor pointers first, coordinate deltas as fallback), and accounts for confusion cells mid-path by pre-inverting subsequent actions.

**Proactive fire avoidance:** Before adding a step for a cell that is in `fire_map`, the agent checks whether that cell will be safe both on the step *into* it and on the following step. If either would be on fire, traversal stops early and the remaining slots are filled with `WAIT`.

**Effect of `last_result` on this state:**

| Condition | Action |
|---|---|
| `last_result.is_dead` | Records the last planned cell as a fire cell (if any were planned). Rebuilds `current_path` from start back to the original destination and re-enters `_traverse`. Remains in `TRAVERSING`. |
| `current_path` reaches length 1 (destination arrived) | Inside `_traverse`, `expanding_cell` is set to the destination, which is added to `visited`. Transitions to `EXPANDING`. |
| `current_path` empty but goal not yet reached | `_pop_and_traverse` pops the best cell from the open queue and builds a new path. |

---

### `RECOVERING`

**Entered from:** `EXPANDING` (fire hit during a probe or a pre-check triggered diversion), `RECOVERING` itself (fire hit again during recovery traversal).

**Purpose:** When fire kills the agent during expansion, the agent respawns at start with no `current_path`. `RECOVERING` navigates the agent back to `expanding_cell` so expansion can resume from where it left off.

**Sub-phases (controlled by `_fire_check_pending` and `_skip_fire_recheck`):**

#### Phase 1 — Traversal back to `expanding_cell`

`_traverse_recovery_path` moves along `_recovery_path` using the same fire-avoidance checks as normal traversal. If the agent dies again along the way, the last planned cell is recorded in `fire_map`, the path is rebuilt from start, and traversal continues.

Remains in `RECOVERING` until `_recovery_path` is reduced to length 1 (arrived).

#### Phase 2 — Re-probe gate (arrived at `expanding_cell`, `_fire_check_pending == False`)

Before re-probing the fire cell the agent waits for a safe window. Safety requires:
- `fire_counter + 1` is not a fire phase (safe to step in).
- `fire_counter + 2` is not a fire phase (safe to return), unless only one safe counter remains (then the +2 requirement is relaxed so the final phase can be sampled and the cell marked impassable rather than waiting forever).
- If no two consecutive safe counters exist, the cell can never be probed safely and `_skip_fire_recheck` is set immediately.

If the gate passes, the probe action is submitted and `_fire_check_pending` is set to `True`.

#### Phase 3 — Re-probe result (`_fire_check_pending == True`)

The agent was at `expanding_cell` and just probed the fire direction.

| Condition | Action |
|---|---|
| `last_result.is_dead` | Fire still present. Records this phase in `fire_map`. If all four phases are now recorded (impassable), sets `_skip_fire_recheck = True`. Rebuilds recovery path and loops. |
| agent moved (`wall_hits == 0`) and no fire | Cell is passable at this phase. Registers it in memory and the open queue. Resumes expansion from the next direction (or transitions to `TRAVERSING` if the fire direction was the last one). |

#### Skip path (`_skip_fire_recheck == True`)

When fire is impassable, upon arriving back at `expanding_cell` the agent skips the re-probe entirely and proceeds directly to the next expansion direction.

---

### `TELEPORT_RETURNING`

**Entered from:** `EXPANDING` when a probe causes a teleport.

**Purpose:** Exploration probes are probe-and-return: move into a neighbor, then move back. When the move lands on a teleport pad, the agent ends up at an unrelated location and must find its way back to `expanding_cell` to continue the expansion.

The return strategy is **two turns per exit attempt**:

- **Turn A** — Try stepping off the teleport destination in one direction (`_teleport_exit_order[_teleport_exit_tried]`). Set `_teleport_stepping_back = True`.
- **Turn B** — Step back in the opposite direction. If the destination cell was the paired teleport pad, this reverse step triggers another teleport back to the original pad (successfully returning). If not, the agent is back where it started.

**Effect of `last_result`:**

| Condition | Action |
|---|---|
| `_teleport_stepping_back == True` and `last_result.wall_hits > 0` | Step-off was blocked by a wall; increment `_teleport_exit_tried`, clear flag, retry Turn A. |
| `_teleport_stepping_back == True` and `last_result.is_confused` | Stepped onto a confusion cell; records it in `confusion_cells`. The return action must be sent un-inverted because the confusion effect will invert it to the correct direction. |
| `_teleport_stepping_back == True` (normal) | Issue the reverse step action. Clear the flag. |
| `last_result.teleported` and `_teleport_stepping_back == False` | Successful return: the reverse step landed on a teleport pad and sent the agent back to the original teleport pad. Resets all teleport state and resumes `EXPANDING` from the next direction. |
| `last_result.teleported` and `_teleport_stepping_back == True` | Accidental double-teleport during Turn A; increment `_teleport_exit_tried` and retry Turn A. |
| `_teleport_exit_tried >= 4` | All exits blocked; issue five `WAIT` actions. |

---

## Helper Methods

### `_expand(last_result) -> list[Action]`

Generates the 5-action sequence for one probe step. Each step is: return from previous probe (if it was open) + probe next direction + fill remaining slots with `WAIT`. Handles the confusion-cell case by pre-inverting the probe action when `expanding_cell` is a confusion cell and the agent is returning through it.

### `_traverse(premade_actions) -> list[Action]`

Fills up to 5 action slots by walking `current_path`. If `premade_actions` is provided (e.g., a return step already decided), those are prepended. Stops early if the next cell is in a dangerous fire window. On arrival at the destination, immediately transitions `plan_state` to `EXPANDING`.

### `_traverse_recovery_path() -> list[Action]`

Simplified traversal that follows `_recovery_path` using only coordinate deltas (no teleport edge lookup). Applies the same fire-window check.

### `_get_route_to_cell(start, destination) -> list[CellPosition]`

Reconstructs a path between two arbitrary memory cells using their parent pointers. Walks up both cells' ancestor chains to find their lowest common ancestor, then concatenates the two partial paths. Returns a list of positions from `start` to `destination`.

### `_register_neighbor(new_cell, direction)`

Wires bidirectional neighbor pointers between a newly discovered cell and `expanding_cell`, based on which expansion direction was being probed.

### `_pop_and_traverse(premade_actions) -> list[Action]`

Pops the highest-priority (lowest `cost + manhattan_distance`) unexpanded cell from the open queue, builds a path to it, and starts `_traverse`. If the queue is empty and the goal is already in memory, routes directly to the goal instead.

### `_probe_cell_pos(direction) -> CellPosition`

Returns the expected coordinate of the cell in `direction` from `expanding_cell`. Used to identify which cell caused a fire death or which cell to wait for in the re-probe gate.

---

## `reset_episode()`

Called between episodes. Persists accumulated knowledge:
- `memory`, `fire_map`, `confusion_cells`, `fire_counter` are **retained**.
- `expanding_cell` (if non-null) is pushed back onto the open queue so it will be re-visited.
- `visited` is rebuilt to include only fully-expanded cells.
- `open_queue` is rebuilt from all non-fully-expanded memory cells.
- All episode-local state (`_recovery_path`, teleport fields, `_last_turn_planned_cells`, etc.) is reset.
- `plan_state` returns to `START`.

---

## `ManualMovementAgent(Agent)`

A thin wrapper for keyboard-driven play. Stores a single pending `next_action`.

#### `set_action(action: Action)`
Sets the next action to execute.

#### `plan_turn(_) -> list[Action]`
Returns `[next_action]` and resets `next_action` to `WAIT`.

---

## Module-Level Helpers

### `_find_fire_groups(graph) -> list[dict]`

Scans the maze graph for `HAZARD` cells and clusters them into connected groups using a flood-fill (8-directional adjacency). For each cluster it identifies the **V-center**: the cell that has exactly two neighbors within the cluster that are not collinear (i.e., forms the vertex of an L or V shape). The center is always on fire; the arms rotate around it.

Returns a list of dicts, each with:
- `center: tuple[int, int]` — the pivot cell
- `arms: list[tuple[int, int]]` — offset vectors of the two arm cells relative to center

---

## Module Constants

| Name | Description |
|---|---|
| `_MAZE_FILES` | Maps `maze_id` string to PNG filename |
| `_ACTION_TO_NEIGHBOR` | Maps movement `Action` to the `MazeCell` attribute name for that neighbor |
| `_CONFUSION_OPPOSITE` | Maps each `Action` to its directional inverse (used by the environment and agent) |
| `_TELEPORT_STATES` | Set of `CellState` values that indicate a teleport pad |
| `_EXPANSION_OFFSETS` | Maps `AStarAgentExpansionState` to `(dx, dy)` coordinate offset |
| `_EXPANSION_ORDER` | Canonical probe order: `[UP, RIGHT, DOWN, LEFT]` |
