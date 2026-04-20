"""
Training runner for the DQN maze agent.

Usage examples
--------------
# Train 2000 episodes on procedurally generated mazes (recommended):
    python train_rl.py --generated --seeds 42 123 999 --episodes 3000

# Train on the real training maze:
    python train_rl.py

# Resume from a saved checkpoint:
    python train_rl.py --load model.pt --episodes 1000

# Run one greedy evaluation episode after training:
    python train_rl.py --episodes 0 --load model.pt --eval
"""
from __future__ import annotations

import argparse
import time

from maze_generator import GridMazeEnvironment, generate_maze
from maze_solver import AStarMazeEnvironment
from dqn_agent import DQNAgent


def run_episode(
    env,
    agent: DQNAgent,
    max_turns: int = 10_000,
) -> dict:
    """Run one episode; return the environment's stats dict."""
    result = None
    for _ in range(max_turns):
        actions = agent.plan_turn(result)
        result = env.step(actions)
        if result.is_goal_reached:
            break
    return env.get_episode_stats()


def train(
    num_episodes: int = 2000,
    use_generated_mazes: bool = False,
    maze_seeds: list[int] | None = None,
    save_path: str = 'model.pt',
    load_path: str | None = None,
    log_interval: int = 10,
    max_turns: int | None = None,  # None = size*size*50, auto-scales with curriculum
    gamma: float = 0.95,
    lr: float = 1e-3,
    epsilon: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    batch_size: int = 128,
    train_start: int = 1_000,
    target_sync: int = 1_000,
    train_freq: int = 4,
    device: str = 'auto',
    start_size: int = 8,
    promote_after: int = 5,
) -> list[dict]:

    agent = DQNAgent(
        gamma=gamma,
        lr=lr,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        train_start=train_start,
        target_sync=target_sync,
        train_freq=train_freq,
        device=device,
    )

    if load_path is not None:
        agent.load(load_path)
        print(f"Loaded model from {load_path}  (epsilon={agent.epsilon:.4f})")

    print(f"Device: {agent.device}")

    if maze_seeds is None:
        maze_seeds = [42, 123, 999]

    stats_history: list[dict] = []
    real_env = AStarMazeEnvironment('training') if not use_generated_mazes else None

    current_size    = start_size
    successes_at_sz = 0

    t0 = time.perf_counter()

    try:
        for ep in range(num_episodes):
            if use_generated_mazes:
                seed   = maze_seeds[ep % len(maze_seeds)]
                size   = current_size
                n_fire = 0 # max(0, (size - 8) // 6)
                n_conf = 0 # max(0, (size - 12) // 8)
                grid   = generate_maze(seed=seed, size=size,
                                       num_fire_hazards=n_fire,
                                       num_confusion=n_conf)
                env    = GridMazeEnvironment(grid)
            else:
                env  = real_env
                size = None

            start_pos = env.reset()
            agent._start_pos = start_pos
            agent._fire_turn = env._fire_turn
            agent.reset_episode()

            turns = max_turns if max_turns is not None else (size * size * 50 if size else 20_000)
            ep_stats = run_episode(env, agent, max_turns=turns)
            ep_stats['epsilon'] = agent.epsilon
            ep_stats['replay']  = len(agent._replay)
            ep_stats['size']    = size
            stats_history.append(ep_stats)

            # Curriculum advancement
            if use_generated_mazes and ep_stats['goal_reached']:
                successes_at_sz += 1
                if successes_at_sz >= promote_after and current_size < 64:
                    current_size    = min(current_size + 4, 64)
                    successes_at_sz = 0
                    elapsed = time.perf_counter() - t0
                    print(f"  ↑ Promoted to size {current_size}  (ep {ep+1}, {elapsed:.1f}s)")

            if (ep + 1) % log_interval == 0:
                recent     = stats_history[-log_interval:]
                avg_turns  = sum(s['turns_taken'] for s in recent) / log_interval
                goal_rate  = sum(1 for s in recent if s['goal_reached'])  / log_interval
                avg_deaths = sum(s['deaths']      for s in recent) / log_interval
                elapsed    = time.perf_counter() - t0
                size_str   = f"Size={current_size:3d} | " if size is not None else ""
                print(
                    f"Ep {ep+1:5d} | "
                    f"{size_str}"
                    f"GoalRate={goal_rate:5.1%} | "
                    f"AvgTurns={avg_turns:7.1f} | "
                    f"AvgDeaths={avg_deaths:5.2f} | "
                    f"Epsilon={agent.epsilon:.4f} | "
                    f"Replay={len(agent._replay):6d} | "
                    f"Elapsed={elapsed:6.1f}s"
                )

    except KeyboardInterrupt:
        print(f"\nInterrupted — saving to {save_path} ...")

    if num_episodes > 0:
        agent.save(save_path)
        print(f"\nTraining complete — model saved to {save_path}")
        print(f"  Final epsilon  : {agent.epsilon:.4f}")
        print(f"  Training steps : {agent._steps}")

    return stats_history


def evaluate(
    load_path: str,
    use_generated: bool = False,
    seed: int = 0,
    size: int = 32,
    max_turns: int | None = None,
) -> None:
    """Run one greedy episode (epsilon=0) and print the result."""
    agent = DQNAgent()
    agent.load(load_path)
    agent.epsilon = 0.0

    if use_generated:
        n_fire = 0 # max(0, (size - 8) // 6)
        n_conf = 0 # max(0, (size - 12) // 8)
        grid = generate_maze(seed=seed, size=size,
                             num_fire_hazards=n_fire, num_confusion=n_conf)
        env = GridMazeEnvironment(grid)
        print(f"Evaluating on generated maze (seed={seed}, size={size}, fire={n_fire}, confusion={n_conf}) ...")
    else:
        env = AStarMazeEnvironment('training')
        print("Evaluating on training maze ...")

    start_pos = env.reset()
    agent._start_pos = start_pos
    agent._fire_turn = env._fire_turn
    agent.reset_episode()

    turns = max_turns if max_turns is not None else size * size * 50
    stats = run_episode(env, agent, max_turns=turns)
    print(f"  Goal reached  : {stats['goal_reached']}")
    print(f"  Turns taken   : {stats['turns_taken']}")
    print(f"  Deaths        : {stats['deaths']}")
    print(f"  Cells explored: {stats['cells_explored']}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the DQN maze agent.")
    p.add_argument('--episodes',      type=int,   default=2000)
    p.add_argument('--generated',     action='store_true',
                   help='Train on procedurally generated mazes')
    p.add_argument('--seeds',         type=int,   nargs='+', default=[42, 123, 999])
    p.add_argument('--save',          type=str,   default='model.pt')
    p.add_argument('--load',          type=str,   default=None)
    p.add_argument('--log-interval',  type=int,   default=10)
    p.add_argument('--max-turns',     type=int,   default=None,
                   help='Max turns per episode (default: size*size*50 for generated, 20000 for real)')
    p.add_argument('--gamma',         type=float, default=0.95)
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--epsilon',       type=float, default=1.0)
    p.add_argument('--epsilon-min',   type=float, default=0.05)
    p.add_argument('--epsilon-decay', type=float, default=0.995)
    p.add_argument('--batch-size',    type=int,   default=128)
    p.add_argument('--train-start',   type=int,   default=1000,
                   help='Replay buffer size before training begins')
    p.add_argument('--target-sync',   type=int,   default=1000,
                   help='Steps between target network syncs')
    p.add_argument('--train-freq',    type=int,   default=4,
                   help='Train every N steps (default: 4)')
    p.add_argument('--device',        type=str,   default='auto',
                   help='Torch device: auto, cpu, cuda (default: auto)')
    p.add_argument('--start-size',    type=int,   default=8,
                   help='Starting maze size for curriculum (default: 8)')
    p.add_argument('--promote-after', type=int,   default=5,
                   help='Successful episodes before advancing maze size by +4 (default: 5)')
    p.add_argument('--eval',          action='store_true')
    p.add_argument('--eval-seed',     type=int,   default=0)
    p.add_argument('--eval-size',     type=int,   default=32,
                   help='Maze size for --eval --generated (default: 32)')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    train(
        num_episodes=args.episodes,
        use_generated_mazes=args.generated,
        maze_seeds=args.seeds,
        save_path=args.save,
        load_path=args.load,
        log_interval=args.log_interval,
        max_turns=args.max_turns,
        gamma=args.gamma,
        lr=args.lr,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        train_start=args.train_start,
        target_sync=args.target_sync,
        train_freq=args.train_freq,
        device=args.device,
        start_size=args.start_size,
        promote_after=args.promote_after,
    )

    if args.eval:
        eval_path = args.save if args.load is None else args.load
        evaluate(
            load_path=eval_path,
            use_generated=args.generated,
            seed=args.eval_seed,
            size=args.eval_size,
            max_turns=args.max_turns,
        )
