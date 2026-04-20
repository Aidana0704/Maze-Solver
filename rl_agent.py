from __future__ import annotations
import pickle
import random
from collections import defaultdict
from typing import override

import numpy as np

from maze_solver import Action, Agent, TurnResult

_ACTIONS = [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT]

# (dx, dy) for each action index: x=col increases right, y=row increases down
_ACTION_DELTAS: list[tuple[int, int]] = [(0, -1), (0, 1), (-1, 0), (1, 0)]

# Reward constants
_R_GOAL      =  500.0
_R_DEATH     = -100.0
_R_WALL      =   -5.0
_R_CONFUSION =  -10.0
_R_NEW_CELL  =    5.0
_R_STEP      =   -1.0


class QLearningAgent(Agent):
    """
    Tabular Q-Learning agent for blind maze navigation.

    State: (goal_dx, goal_dy, wall_bits, fire_phase)
      - goal_dx, goal_dy: sign of (goal - current) in each axis → 9 combinations.
        Relative, so the same state means the same thing in any maze.
      - wall_bits: 4-tuple of bools encoding which of (UP,DOWN,LEFT,RIGHT) are
        known to be blocked at the current cell, learned from wall hits this episode.
        Resets each episode so stale maze-specific data never carries over.
      - fire_phase: fire_turn % 4 — which V-arm rotation is active.

    Total state space: 9 × 16 × 4 = 576 states.

    Submits exactly 1 action per turn for a clean per-step Bellman update.
    Only the Q-table and remembered goal position persist across episodes;
    everything else (wall memory, fire danger, danger cells) resets each
    episode so the agent generalises across different maze layouts.
    """

    @override
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # ── Cross-episode persistent state ─────────────────────────────────
        # Q-table: state → [Q_UP, Q_DOWN, Q_LEFT, Q_RIGHT]
        self.q_table: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(4))

        # Remembered goal position — set when goal is first reached, persists
        # across episodes so the agent can orient toward it from the start.
        self._goal_pos: tuple[int, int] | None = None

        # Mirrors the environment's _fire_turn — synced by the training loop
        # at the start of each episode; never reset here.
        self._fire_turn: int = 0

        # ── Episode-local state (reset by reset_episode) ────────────────────
        self._start_pos: tuple[int, int] = (0, 0)
        self._current_pos: tuple[int, int] = (0, 0)
        self._prev_state: tuple | None = None
        self._prev_action_idx: int | None = None
        self._intended_next_pos: tuple[int, int] | None = None
        self._episode_visited: set[tuple[int, int]] = set()

        # Wall memory: maps position → set of action indices known to be blocked.
        # Built from observed wall hits; reset each episode so it reflects only
        # the current maze layout.
        self._wall_memory: dict[tuple[int, int], set[int]] = {}

        # Fire / danger tracking — position-specific, reset each episode so
        # data from a previous maze doesn't corrupt decisions in a new one.
        self.fire_danger: dict[tuple[int, int], set[int]] = {}
        self.danger_cells: set[tuple[int, int]] = set()
        self.confusion_cells: set[tuple[int, int]] = set()

    # ── Internal helpers ────────────────────────────────────────────────────

    def _make_state(self) -> tuple:
        x, y = self._current_pos

        # Direction to goal: relative, maze-agnostic
        if self._goal_pos is not None:
            gx, gy = self._goal_pos
            goal_dx = 0 if gx == x else (1 if gx > x else -1)
            goal_dy = 0 if gy == y else (1 if gy > y else -1)
        else:
            goal_dx, goal_dy = 0, 0  # goal not yet found

        # Known wall pattern at current cell (4-bit: UP, DOWN, LEFT, RIGHT)
        blocked = self._wall_memory.get((x, y), frozenset())
        wall_bits = tuple(i in blocked for i in range(4))

        return (goal_dx, goal_dy, wall_bits, self._fire_turn % 4)

    def _select_action(self, state: tuple) -> int:
        if random.random() < self.epsilon:
            return random.randrange(4)

        q = self.q_table[state].copy()
        x, y = self._current_pos
        fire_mod = self._fire_turn % 4

        for i, (dx, dy) in enumerate(_ACTION_DELTAS):
            nb = (x + dx, y + dy)
            if nb in self.danger_cells:
                q[i] = -np.inf
            elif nb in self.fire_danger and fire_mod in self.fire_danger[nb]:
                q[i] = -np.inf

        if np.all(q == -np.inf):
            return random.randrange(4)
        return int(np.argmax(q))

    def _mark_danger(self, pos: tuple[int, int], fire_mod: int) -> None:
        self.fire_danger.setdefault(pos, set()).add(fire_mod)
        if len(self.fire_danger[pos]) == 4:
            self.danger_cells.add(pos)

    def _q_update(
        self,
        prev_state: tuple,
        action_idx: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ) -> None:
        old = self.q_table[prev_state][action_idx]
        target = reward if done else reward + self.gamma * float(np.max(self.q_table[next_state]))
        self.q_table[prev_state][action_idx] += self.alpha * (target - old)

    # ── Agent interface ─────────────────────────────────────────────────────

    @override
    def plan_turn(self, last_result: TurnResult | None) -> list[Action]:
        if last_result is None:
            self._current_pos = self._start_pos
            state = self._make_state()
            action_idx = self._select_action(state)
            self._prev_state = state
            self._prev_action_idx = action_idx
            dx, dy = _ACTION_DELTAS[action_idx]
            self._intended_next_pos = (
                self._current_pos[0] + dx,
                self._current_pos[1] + dy,
            )
            self._fire_turn += 1
            return [_ACTIONS[action_idx]]

        # ── Process feedback from last step ────────────────────────────────
        new_pos: tuple[int, int] = last_result.current_position
        died     = last_result.is_dead
        goal     = last_result.is_goal_reached
        wall_hit = last_result.wall_hits > 0
        confused = last_result.is_confused

        fire_mod_at_step = self._fire_turn % 4

        if died and self._intended_next_pos is not None:
            self._mark_danger(self._intended_next_pos, fire_mod_at_step)

        if confused:
            self.confusion_cells.add(new_pos)

        # Record blocked direction at current cell when a wall was hit
        if wall_hit and self._prev_action_idx is not None:
            self._wall_memory.setdefault(self._current_pos, set()).add(
                self._prev_action_idx
            )

        # Remember goal position when first found
        if goal:
            self._goal_pos = new_pos

        # ── Compute reward ──────────────────────────────────────────────────
        reward = _R_STEP
        if goal:
            reward += _R_GOAL
        if died:
            reward += _R_DEATH
        if wall_hit:
            reward += _R_WALL
        if confused:
            reward += _R_CONFUSION
        if new_pos not in self._episode_visited:
            reward += _R_NEW_CELL
            self._episode_visited.add(new_pos)

        # ── Update position, build next state ──────────────────────────────
        self._current_pos = new_pos
        next_state = self._make_state()

        # ── Bellman update ──────────────────────────────────────────────────
        if self._prev_state is not None:
            self._q_update(
                self._prev_state, self._prev_action_idx,
                reward, next_state, done=goal,
            )

        if goal:
            return [Action.WAIT]

        # ── Select next action ──────────────────────────────────────────────
        action_idx = self._select_action(next_state)
        self._prev_state = next_state
        self._prev_action_idx = action_idx
        dx, dy = _ACTION_DELTAS[action_idx]
        self._intended_next_pos = (
            self._current_pos[0] + dx,
            self._current_pos[1] + dy,
        )
        self._fire_turn += 1
        return [_ACTIONS[action_idx]]

    @override
    def reset_episode(self) -> None:
        self._current_pos       = self._start_pos
        self._prev_state        = None
        self._prev_action_idx   = None
        self._intended_next_pos = None
        self._episode_visited   = set()
        self._wall_memory       = {}
        self.fire_danger        = {}
        self.danger_cells       = set()
        self.confusion_cells    = set()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        data = {
            'q_table':    dict(self.q_table),
            '_goal_pos':  self._goal_pos,
            'epsilon':    self.epsilon,
            '_fire_turn': self._fire_turn,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.q_table    = defaultdict(lambda: np.zeros(4), data['q_table'])
        self._goal_pos  = data.get('_goal_pos')
        self.epsilon    = data['epsilon']
        self._fire_turn = data['_fire_turn']
