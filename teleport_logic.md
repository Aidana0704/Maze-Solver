# Teleport Logic in `AStarAgent`

## Background

Teleport pads move the agent to an unknown destination the moment it steps on one.
This breaks the expansion loop because the agent is no longer standing next to the cell it was probing from (`expanding_cell`).
The agent's recovery strategy has two phases:

1. **Discovery** – detect the teleport during expansion and record the destination cell.
2. **Returning** – find a way back to the original teleporter, then walk home.

---

## Phase 1 – Discovery (inside `EXPANDING` state)

During a normal probe, the agent issues a single-step move.  
If `last_result.teleported` comes back True, the agent has been moved to a new, possibly far-away cell.

What happens:
- The destination position is stored in memory as a new `AStarAgentMemoryCell` linked back to `expanding_cell` via the probe direction.
- The destination is pushed onto the A\* open queue so it will be explored later.
- `_teleport_origin_dir` is recorded — this is the expansion direction that was mid-flight (UP / RIGHT / DOWN / LEFT).
- All return-state variables are reset (`_teleport_exit_tried = 0`, `_teleport_stepping_back = False`).
- Planning state switches to `TELEPORT_RETURNING`.
- The agent returns `WAIT × 5` this turn because it needs a full turn to re-orient.

---

## Phase 2 – Returning (`TELEPORT_RETURNING` state)

The agent is now sitting on the teleport destination cell.
The insight the code relies on: **stepping back onto the destination should re-teleport the agent to the original teleporter**, which is adjacent to `expanding_cell`.

The sequence of moves is called **Turn A → Turn B**:

| Turn | What the agent does |
|------|---------------------|
| A | Steps *off* the destination in some direction. |
| B | Steps *back* onto the destination, which should teleport back to origin. |

There are four branches checked in order each planning cycle:

---

### Branch 1 — Accidental re-teleport during Turn A step-off

**Condition:** `last_result.teleported == True` **and** `_teleport_stepping_back == True`

The step-off in Turn A itself landed on *another* teleporter, so the agent was carried somewhere new instead of just stepping to an adjacent cell.

Recovery:
- Increment `_teleport_exit_tried` to abandon this exit direction.
- Reset `_teleport_stepping_back = False`.
- Fall through to Turn A (below) to try the next direction.

---

### Branch 2 — Successfully returned to the origin teleporter

**Condition:** `last_result.teleported == True` **and** `_teleport_stepping_back == False`

Turn B's step-back landed on the destination cell and the destination teleported the agent back to the original teleporter.
The agent is now adjacent to `expanding_cell`.

Recovery:
- Compute the return action (opposite of `_teleport_origin_dir`) to step back onto `expanding_cell`.
- Reset all teleport state (`_teleport_in_progress`, `_teleport_origin_dir`, `_teleport_exit_tried`, etc.).
- Look up the index of `_teleport_origin_dir` in the four-direction expansion order `[UP, RIGHT, DOWN, LEFT]`:
  - **If LEFT was the origin direction** (last direction in order): the probe was the final one. Mark `expanding_cell.fully_expanded = True` and call `_pop_and_traverse([ret_action])` — the return step is bundled into the traversal turn.
  - **Otherwise**: transition back to `EXPANDING` and issue `[ret_action, next_probe_action] + WAIT × 3` — return home *and* immediately start the next probe direction in the same turn.

---

### Branch 3 — Turn B (stepping back onto the destination)

**Condition:** `_teleport_stepping_back == True` (Turn A was sent last turn)

The agent already stepped off the destination. Now it needs to step back on.

Three sub-cases depending on what Turn A's result shows:

#### 3a — Exit was blocked (hit a wall)
`last_result.wall_hits > 0`

The agent never moved. Increment `_teleport_exit_tried` to try the next direction and reset `_teleport_stepping_back = False`.  
(Falls through to Turn A next cycle.)

#### 3b — Stepped onto a confusion cell
`last_result.is_confused == True`

The agent moved but landed on a confusion tile, which means future actions in that turn are reversed.  
The step-back must be sent as the **same direction** as the exit — confusion will invert it into the correct opposite.  
The confusion cell position is recorded in `confusion_cells`.

Result: `[_teleport_last_exit] + WAIT × 4`, `_teleport_stepping_back = False`.

#### 3c — Normal step-off succeeded
Default case.

Compute the reverse of the exit direction with `_CONFUSION_OPPOSITE` and send that as the step-back.

Result: `[reverse_of_exit] + WAIT × 4`, `_teleport_stepping_back = False`.

---

### Branch 4 — Turn A (trying to step off the destination)

**Condition:** Default — no teleport event and not currently mid-step-back.

Try each exit direction from `_teleport_exit_order = [UP, DOWN, LEFT, RIGHT]` in sequence.

- **All directions exhausted** (`_teleport_exit_tried >= 4`): every neighbor is a wall. The agent is stuck on the destination; return `WAIT × 5`.
- **Otherwise**: pick `_teleport_exit_order[_teleport_exit_tried]`, store it in `_teleport_last_exit`, set `_teleport_stepping_back = True`, and return `[exit_action] + WAIT × 4`.

---

## State variable cheat-sheet

| Variable | Meaning |
|----------|---------|
| `_teleport_origin_dir` | Which expansion direction (UP/RIGHT/DOWN/LEFT) triggered the teleport |
| `_teleport_exit_tried` | How many exit directions have been attempted from the destination |
| `_teleport_last_exit` | The specific `Action` used in the most recent Turn A |
| `_teleport_stepping_back` | `True` when the agent has stepped off and is about to step back (Turn B) |

---

## Known edge cases / limitations

- **Destination has no open neighbors**: the agent will exhaust all four directions, hit walls each time, and then wait forever on `WAIT × 5`. There is no escape from a completely walled-in destination.
- **Turn A lands on a second teleporter**: handled by Branch 1 (skip that direction), but if every direction leads to another teleporter the agent will cycle through them and never find a safe exit.
- **Re-teleport goes to the wrong place**: Branch 2 assumes that stepping back onto the destination always returns the agent to the original teleporter. If a teleporter is one-way (destination ≠ inverse of origin), the agent will treat wherever it ends up as the origin and resume expansion incorrectly.
