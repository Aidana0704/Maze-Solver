from __future__ import annotations

import random
from typing import override

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from maze_solver import Action, Agent, TurnResult

_ACTIONS = [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT]
_ACTION_DELTAS: list[tuple[int, int]] = [(0, -1), (0, 1), (-1, 0), (1, 0)]
_ACTION_OPPOSITES = [1, 0, 3, 2]  # N↔S, W↔E

_R_GOAL      =  500.0
_R_DEATH     = -100.0
_R_WALL      =   -5.0
_R_CONFUSION =  -10.0
_R_NEW_CELL  =    5.0
_R_STEP      =   -1.0

MAP_SIZE     = 64    # spatial obs: MAP_SIZE×MAP_SIZE pixels anchored to start_pos
N_SPATIAL_CH = 9     # unknown | wall_N S W E | fire | confusion | agent | goal
N_GLOBAL     = 4     # fire-phase one-hot

_SPATIAL_FLAT = N_SPATIAL_CH * MAP_SIZE * MAP_SIZE   # 36 864
_OBS_DIM      = _SPATIAL_FLAT + N_GLOBAL             # 36 868

# Numpy world-map dimensions — positions map as: array[wy+_OFF, wx+_OFF]
_MAP = 512
_OFF = 256


class _QNet(nn.Module):
    """CNN that processes a MAP_SIZE×MAP_SIZE×N_SPATIAL_CH image + N_GLOBAL scalars."""
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(N_SPATIAL_CH, 32, 3, stride=2, padding=1),   # (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),              # (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),             # (128, 8, 8)
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8 + N_GLOBAL, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, spatial: torch.Tensor, global_: torch.Tensor) -> torch.Tensor:
        # spatial: (B, N_SPATIAL_CH, MAP_SIZE, MAP_SIZE)  global_: (B, N_GLOBAL)
        x = self.cnn(spatial).reshape(spatial.size(0), -1)
        return self.fc(torch.cat([x, global_], dim=1))


class _ReplayBuffer:
    """
    Circular replay buffer with uint8 spatial storage.

    Spatial values are in {0, 0.5, 1.0}; stored as uint8 {0, 1, 2} (×2) and
    restored as /2.0 on sample.  This cuts buffer RAM ~4× vs float32.
    At 50 k capacity: ~1.8 GB;  at 100 k: ~3.7 GB.
    """
    def __init__(self, capacity: int):
        self._buf  = [None] * capacity
        self._cap  = capacity
        self._idx  = 0
        self._size = 0

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        self._buf[self._idx] = (
            (s [:_SPATIAL_FLAT] * 2).astype(np.uint8),  # spatial ×2 → uint8
            s [_SPATIAL_FLAT:],                          # global  float32
            a, r,
            (s2[:_SPATIAL_FLAT] * 2).astype(np.uint8),
            s2[_SPATIAL_FLAT:],
            done,
        )
        self._idx  = (self._idx + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def sample(self, n: int):
        idxs  = random.sample(range(self._size), n)
        batch = [self._buf[i] for i in idxs]
        sp, sg, a, r, sp2, sg2, d = zip(*batch)
        s  = np.concatenate([np.array(sp,  dtype=np.float32) / 2.0,
                              np.array(sg,  dtype=np.float32)], axis=1)
        s2 = np.concatenate([np.array(sp2, dtype=np.float32) / 2.0,
                              np.array(sg2, dtype=np.float32)], axis=1)
        return (
            s,
            np.array(a,  dtype=np.int64),
            np.array(r,  dtype=np.float32),
            s2,
            np.array(d,  dtype=np.float32),
        )

    def __len__(self) -> int:
        return self._size


class DQNAgent(Agent):
    """
    DQN agent for blind maze navigation.

    Observation: a MAP_SIZE×MAP_SIZE global map (9 channels) anchored to
    start_pos, plus a 4-dim fire-phase one-hot.  The CNN sees the full explored
    map — unlike a local patch it can plan around previously-seen walls.

    Channel layout (9 spatial):
        0  unknown      1=unseen  0=visited
        1  wall_N       0=passable  0.5=untested  1.0=blocked
        2  wall_S
        3  wall_W
        4  wall_E
        5  fire         1=known fire cell
        6  confusion    1=known confusion cell
        7  agent_pos    1 at current cell only
        8  goal_pos     1 at goal once discovered

    All map memory resets each episode; network weights are the only thing that
    persists across mazes, forcing the policy to generalise.

    Replay buffer stores spatial as uint8 to keep RAM usage reasonable.
    """

    def __init__(
        self,
        gamma: float = 0.95,
        lr: float = 1e-3,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 128,
        replay_capacity: int = 50_000,
        target_sync: int = 1_000,
        train_start: int = 1_000,
        train_freq: int = 4,
        device: str = 'auto',
    ):
        super().__init__()
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_sync   = target_sync
        self.train_start   = train_start
        self.train_freq    = train_freq

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self._online = _QNet().to(self.device)
        self._target = _QNet().to(self.device)
        self._target.load_state_dict(self._online.state_dict())
        self._target.eval()
        self._opt  = optim.Adam(self._online.parameters(), lr=lr)
        self._loss = nn.SmoothL1Loss()

        self._replay = _ReplayBuffer(replay_capacity)
        self._steps  = 0

        # ── Cross-episode persistent state ─────────────────────────────────────
        self._fire_turn: int = 0

        # ── Numpy map — reset each episode via .fill() ─────────────────────────
        # _map_unknown:  1.0=unseen  0.0=visited
        # _map_walls:    per-direction  0.0=passable  0.5=untested  1.0=wall
        # _map_ctype:    [fire, confusion, goal]
        self._map_unknown = np.ones( (_MAP, _MAP),    dtype=np.float32)
        self._map_walls   = np.full( (_MAP, _MAP, 4), 0.5, dtype=np.float32)
        self._map_ctype   = np.zeros((_MAP, _MAP, 3), dtype=np.float32)

        # ── Episode-local state ────────────────────────────────────────────────
        self._start_pos:       tuple[int, int] = (0, 0)
        self._current_pos:     tuple[int, int] = (0, 0)
        self._goal_pos:        tuple[int, int] | None = None
        self._prev_obs:        np.ndarray | None = None
        self._prev_action:     int | None = None
        self._intended_next:   tuple[int, int] | None = None
        self._episode_visited: set[tuple[int, int]] = set()
        self._step_counter:    int = 0

    # ── Map helpers ────────────────────────────────────────────────────────────

    def _rc(self, wx: int, wy: int) -> tuple[int, int]:
        return wy + _OFF, wx + _OFF

    def _visit(self, wx: int, wy: int) -> None:
        r, c = self._rc(wx, wy)
        self._map_unknown[r, c] = 0.0

    def _mark_wall(self, wx: int, wy: int, d: int, val: float) -> None:
        r, c = self._rc(wx, wy)
        self._map_walls[r, c, d] = val

    def _mark_ctype(self, wx: int, wy: int, idx: int) -> None:
        r, c = self._rc(wx, wy)
        self._map_ctype[r, c, :] = 0.0
        self._map_ctype[r, c, idx] = 1.0

    # ── Observation ────────────────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        sx, sy = self._start_pos
        cx, cy = self._current_pos
        sr, sc = sy + _OFF, sx + _OFF

        # Extract MAP_SIZE×MAP_SIZE window anchored to start_pos
        unk   = self._map_unknown[sr:sr+MAP_SIZE, sc:sc+MAP_SIZE]       # (64,64)
        walls = self._map_walls  [sr:sr+MAP_SIZE, sc:sc+MAP_SIZE, :]    # (64,64,4)
        ctype = self._map_ctype  [sr:sr+MAP_SIZE, sc:sc+MAP_SIZE, :]    # (64,64,3)

        # Agent position channel
        agent_ch = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
        adx, ady = cx - sx, cy - sy
        if 0 <= adx < MAP_SIZE and 0 <= ady < MAP_SIZE:
            agent_ch[ady, adx] = 1.0

        # Goal position channel (once discovered)
        goal_ch = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
        if self._goal_pos is not None:
            gx, gy = self._goal_pos
            gdx, gdy = gx - sx, gy - sy
            if 0 <= gdx < MAP_SIZE and 0 <= gdy < MAP_SIZE:
                goal_ch[gdy, gdx] = 1.0

        # Stack 9 channels → (9, 64, 64)
        spatial = np.stack([
            unk,
            walls[:, :, 0], walls[:, :, 1], walls[:, :, 2], walls[:, :, 3],
            ctype[:, :, 0], ctype[:, :, 1],
            agent_ch,
            goal_ch,
        ], dtype=np.float32)

        # Fire-phase one-hot reflects current turn parity
        fire_oh = np.zeros(4, dtype=np.float32)
        fire_oh[self._fire_turn % 4] = 1.0

        return np.concatenate([spatial.reshape(-1), fire_oh])

    # ── Action selection ───────────────────────────────────────────────────────

    def _select_action(self, obs: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(4)

        s_t = torch.tensor(
            obs[:_SPATIAL_FLAT].reshape(1, N_SPATIAL_CH, MAP_SIZE, MAP_SIZE),
            dtype=torch.float32, device=self.device,
        )
        g_t = torch.tensor(
            obs[_SPATIAL_FLAT:].reshape(1, N_GLOBAL),
            dtype=torch.float32, device=self.device,
        )
        with torch.no_grad():
            q = self._online(s_t, g_t).squeeze(0).cpu().numpy()

        cx, cy = self._current_pos
        for i, (dx, dy) in enumerate(_ACTION_DELTAS):
            r, c = self._rc(cx + dx, cy + dy)
            if self._map_ctype[r, c, 0] == 1.0:  # known fire — hard-mask
                q[i] = -np.inf

        return random.randrange(4) if np.all(q == -np.inf) else int(np.argmax(q))

    # ── Training step ──────────────────────────────────────────────────────────

    def _train(self) -> None:
        if len(self._replay) < self.train_start:
            return

        s, a, r, s2, d = self._replay.sample(self.batch_size)

        S_sp  = torch.tensor(s [:, :_SPATIAL_FLAT].reshape(-1, N_SPATIAL_CH, MAP_SIZE, MAP_SIZE),
                              dtype=torch.float32, device=self.device)
        S_gl  = torch.tensor(s [:, _SPATIAL_FLAT:], dtype=torch.float32, device=self.device)
        S2_sp = torch.tensor(s2[:, :_SPATIAL_FLAT].reshape(-1, N_SPATIAL_CH, MAP_SIZE, MAP_SIZE),
                              dtype=torch.float32, device=self.device)
        S2_gl = torch.tensor(s2[:, _SPATIAL_FLAT:], dtype=torch.float32, device=self.device)
        A     = torch.tensor(a, dtype=torch.int64,   device=self.device)
        R     = torch.tensor(r, dtype=torch.float32, device=self.device)
        D     = torch.tensor(d, dtype=torch.float32, device=self.device)

        q_pred = self._online(S_sp, S_gl).gather(1, A.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self._target(S2_sp, S2_gl).max(1).values
            q_tgt  = R + self.gamma * q_next * (1.0 - D)

        loss = self._loss(q_pred, q_tgt)
        self._opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._online.parameters(), 10.0)
        self._opt.step()

        self._steps += 1
        if self._steps % self.target_sync == 0:
            self._target.load_state_dict(self._online.state_dict())

    # ── Agent interface ────────────────────────────────────────────────────────

    @override
    def plan_turn(self, last_result: TurnResult | None) -> list[Action]:
        if last_result is None:
            self._current_pos = self._start_pos
            self._visit(*self._current_pos)
            obs = self._build_obs()
            action = self._select_action(obs)
            self._prev_obs    = obs
            self._prev_action = action
            dx, dy = _ACTION_DELTAS[action]
            self._intended_next = (self._current_pos[0] + dx, self._current_pos[1] + dy)
            self._fire_turn += 1
            return [_ACTIONS[action]]

        new_pos  = last_result.current_position
        died     = last_result.is_dead
        goal     = last_result.is_goal_reached
        wall_hit = last_result.wall_hits > 0
        confused = last_result.is_confused
        prev     = self._prev_action

        # ── Update map from last action's outcome ───────────────────────────────
        cx, cy = self._current_pos

        if wall_hit and prev is not None:
            self._mark_wall(cx, cy, prev, 1.0)
        elif prev is not None and not died:
            self._mark_wall(cx, cy, prev, 0.0)
            self._mark_wall(*new_pos, _ACTION_OPPOSITES[prev], 0.0)

        if died and self._intended_next is not None:
            self._mark_ctype(*self._intended_next, 0)   # fire
        if confused:
            self._mark_ctype(*new_pos, 1)               # confusion
        if goal:
            self._mark_ctype(*new_pos, 2)               # goal
            self._goal_pos = new_pos

        self._visit(*new_pos)

        # ── Reward ──────────────────────────────────────────────────────────────
        reward = _R_STEP
        if goal:     reward += _R_GOAL
        if died:     reward += _R_DEATH
        if wall_hit: reward += _R_WALL
        if confused: reward += _R_CONFUSION
        if new_pos not in self._episode_visited:
            reward += _R_NEW_CELL
            self._episode_visited.add(new_pos)

        self._current_pos  = new_pos
        self._step_counter += 1
        next_obs = self._build_obs()

        if self._prev_obs is not None:
            self._replay.push(self._prev_obs, self._prev_action, reward, next_obs, goal)
            if self._step_counter % self.train_freq == 0:
                self._train()

        if goal:
            return [Action.WAIT]

        action = self._select_action(next_obs)
        self._prev_obs    = next_obs
        self._prev_action = action
        dx, dy = _ACTION_DELTAS[action]
        self._intended_next = (self._current_pos[0] + dx, self._current_pos[1] + dy)
        self._fire_turn += 1
        return [_ACTIONS[action]]

    @override
    def reset_episode(self) -> None:
        self._current_pos      = self._start_pos
        self._goal_pos         = None
        self._prev_obs         = None
        self._prev_action      = None
        self._intended_next    = None
        self._episode_visited  = set()
        self._step_counter     = 0
        self._map_unknown.fill(1.0)
        self._map_walls.fill(0.5)
        self._map_ctype.fill(0.0)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            'online':    self._online.state_dict(),
            'target':    self._target.state_dict(),
            'optimizer': self._opt.state_dict(),
            'epsilon':   self.epsilon,
            'steps':     self._steps,
            'fire_turn': self._fire_turn,
        }, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self.device, weights_only=True)
        self._online.load_state_dict(data['online'])
        self._target.load_state_dict(data['target'])
        self._opt.load_state_dict(data['optimizer'])
        self.epsilon    = data['epsilon']
        self._steps     = data['steps']
        self._fire_turn = data.get('fire_turn', 0)
