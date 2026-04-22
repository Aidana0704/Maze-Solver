import argparse
import sys

from pygame import Color
import pygame
from load_image import MazeCell, convert_grid_point_to_image_point, CellState
from maze_solver import AStarAgent, AStarMazeEnvironment, TurnResult, ManualMovementAgent, Agent, Action
from solve_maze import get_start_and_goal_pos
import pygame_gui
from pygame_gui.elements import UIPanel, UILabel

BACKGROUND_COLOR = Color(255, 255, 255)
WALL_COLOR = Color(0, 0, 0)
START_COLOR = Color(0, 0, 255)
GOAL_COLOR = Color(0, 255, 0)
PATH_COLOR = Color(255, 0, 0)
VISITED_COLOR = Color(25, 216, 255)
TRAVERSING_COLOR = Color(255, 157, 0)
HAZARD_COLOR = Color(255, 216, 0)
CONFUSION_COLOR = Color(188, 37, 168)
OPEN_QUEUE_COLOR = Color(0, 255, 255)
EXPANDING_COLOR = Color(255, 0, 0)

CAM_SPEED_PX_PER_SECOND = 300
FRAMES_PER_SECOND = 60
TICKS_PER_SECOND = 300
STEPS_PER_TICK = 30
MAX_STEPS_PER_EPISODE = 100_000

TICK_EVENT = pygame.USEREVENT + 1
CAM_SPEED_PX_PER_FRAME = CAM_SPEED_PX_PER_SECOND // FRAMES_PER_SECOND

LITTLE_GUY = pygame.image.load("lil_guy.png")
FLAME_HAZARD = pygame.image.load("flame_hazard.png")
PURPLE_TELEPORT = pygame.image.load("purple_teleport.png")
GREEN_TELEPORT = pygame.image.load("green_teleport.png")
YELLOW_TELEPORT = pygame.image.load("yellow_teleport.png")

camera_x: int = 0
camera_y: int = 0

pygame.init()
screen = pygame.display.set_mode((800, 800))
pygame.display.set_caption("Maze Solver")
manager = pygame_gui.UIManager((800, 800), 'theme.json')

# ── Debug panel ──────────────────────────────────────────────────────────────
# Sits on the right side of the screen.  Height covers all debug sections.
_PANEL_X = 575
_PANEL_W = 218
_PANEL_H = 560
ui_frame = UIPanel(relative_rect=pygame.Rect(_PANEL_X, 11, _PANEL_W, _PANEL_H),
                   manager=manager)

_LBL_W = 200  # label width inside the panel
_LBL_H = 17   # label height


def _lbl(y: int, text: str = "") -> UILabel:
    return UILabel(relative_rect=pygame.Rect(8, y, _LBL_W, _LBL_H),
                   text=text, manager=manager, container=ui_frame)


def _sec(y: int, title: str) -> UILabel:
    """Static section-header label (never updated)."""
    lbl = UILabel(relative_rect=pygame.Rect(8, y, _LBL_W, _LBL_H),
                  text=f"── {title} ──",
                  manager=manager, container=ui_frame)
    return lbl


# ── GENERAL section ──────────────────────────────────────────────────────────
_sec(4, "GENERAL")
_lbl_paused     = _lbl(22)
_lbl_episode    = _lbl(40)
_lbl_steps      = _lbl(58)
_lbl_memory     = _lbl(76)
_lbl_queue      = _lbl(94)

# ── PLANNING section ─────────────────────────────────────────────────────────
_sec(116, "PLANNING")
_lbl_plan_state  = _lbl(134)
_lbl_exp_state   = _lbl(152)
_lbl_expanding   = _lbl(170)
_lbl_path        = _lbl(188)

# ── LAST RESULT section ──────────────────────────────────────────────────────
_sec(210, "LAST RESULT")
_lbl_res_pos     = _lbl(228)
_lbl_res_hits    = _lbl(246)
_lbl_res_flags   = _lbl(264)
_lbl_actions     = _lbl(282)

# ── TELEPORT section ─────────────────────────────────────────────────────────
_sec(304, "TELEPORT")
_lbl_tele_origin = _lbl(322)
_lbl_tele_tried  = _lbl(340)
_lbl_tele_exit   = _lbl(358)
_lbl_tele_step   = _lbl(376)

# ── FIRE section ─────────────────────────────────────────────────────────────
_sec(398, "FIRE")
_lbl_fire_ctr    = _lbl(416)
_lbl_fire_dir    = _lbl(434)
_lbl_fire_flags  = _lbl(452)

# ── RECOVERY section ─────────────────────────────────────────────────────────
_sec(474, "RECOVERY")
_lbl_rec_path    = _lbl(492)
_lbl_rec_dir     = _lbl(510)
_lbl_rec_flags   = _lbl(528)


def _yn(b: bool) -> str:
    return "Y" if b else "N"


def _pos(p) -> str:
    if p is None:
        return "None"
    return f"({p[0]},{p[1]})" if not hasattr(p, 'x') else f"({p.x},{p.y})"


def _act(a) -> str:
    if a is None:
        return "None"
    name_map = {
        Action.MOVE_UP: "UP", Action.MOVE_DOWN: "DN",
        Action.MOVE_LEFT: "LT", Action.MOVE_RIGHT: "RT",
        Action.WAIT: "WT",
    }
    return name_map.get(a, str(a))


def draw_ui(agent: AStarAgent, last_result: TurnResult, last_actions: list[Action],
            paused: bool, episode_count: int, step_count: int):
    # ── GENERAL ──────────────────────────────────────────────────────────────
    _lbl_paused.set_text(f"Paused: {'Y' if paused else 'N'}")
    _lbl_episode.set_text(f"Episode: {episode_count}")
    _lbl_steps.set_text(f"Steps: {step_count}/{MAX_STEPS_PER_EPISODE}")
    _lbl_memory.set_text(f"Memory: {len(agent.memory)} cells")
    _lbl_queue.set_text(f"Open queue: {len(agent.open_queue)}")

    # ── PLANNING ─────────────────────────────────────────────────────────────
    _lbl_plan_state.set_text(f"Plan: {agent.plan_state.name}")
    _lbl_exp_state.set_text(f"Expand: {agent.expansion_state.name}")
    exp_pos = _pos(agent.expanding_cell.pos) if agent.expanding_cell else "None"
    _lbl_expanding.set_text(f"Cell: {exp_pos}")
    _lbl_path.set_text(f"Path len: {len(agent.current_path)}")

    # ── LAST RESULT ───────────────────────────────────────────────────────────
    _lbl_res_pos.set_text(f"Pos: {_pos(last_result.current_position)}")
    _lbl_res_hits.set_text(
        f"Walls: {last_result.wall_hits}  "
        f"Acts: {last_result.actions_executed}")
    _lbl_res_flags.set_text(
        f"dead={_yn(last_result.is_dead)} "
        f"conf={_yn(last_result.is_confused)} "
        f"tele={_yn(last_result.teleported)} "
        f"goal={_yn(last_result.is_goal_reached)}")
    _lbl_actions.set_text("Acts: " + " ".join(_act(a) for a in last_actions))

    # ── TELEPORT ─────────────────────────────────────────────────────────────
    orig = agent._teleport_origin_dir.name if agent._teleport_origin_dir else "None"
    _lbl_tele_origin.set_text(f"Origin dir: {orig}")
    _lbl_tele_tried.set_text(f"Exit tried: {agent._teleport_exit_tried}/4")
    _lbl_tele_exit.set_text(f"Last exit: {_act(agent._teleport_last_exit)}")
    _lbl_tele_step.set_text(f"Stepping back: {_yn(agent._teleport_stepping_back)}")

    # ── FIRE ─────────────────────────────────────────────────────────────────
    _lbl_fire_ctr.set_text(
        f"Counter: {agent.fire_counter}  "
        f"Known: {len(agent.fire_map)}")
    chk_dir = agent._fire_check_direction.name if agent._fire_check_direction else "None"
    _lbl_fire_dir.set_text(f"Check dir: {chk_dir}")
    _lbl_fire_flags.set_text(
        f"pending={_yn(agent._fire_check_pending)}  "
        f"skip={_yn(agent._skip_fire_recheck)}")

    # ── RECOVERY ─────────────────────────────────────────────────────────────
    _lbl_rec_path.set_text(f"Path len: {len(agent._recovery_path)}")
    nxt = _pos(agent._recovery_path[1]) if len(agent._recovery_path) > 1 else "—"
    _lbl_rec_dir.set_text(f"Next step: {nxt}")
    dst = _pos(agent._recovery_path[-1]) if agent._recovery_path else "—"
    _lbl_rec_flags.set_text(f"Dest: {dst}")


def draw_rl_ui(agent, paused: bool, episode_count: int, step_count: int):
    _lbl_paused.set_text(f"Paused: {'Y' if paused else 'N'}")
    _lbl_episode.set_text(f"Episode: {episode_count}")
    _lbl_steps.set_text(f"Steps: {step_count}/{MAX_STEPS_PER_EPISODE}")
    _lbl_memory.set_text(f"Visited: {len(agent._episode_visited)}")
    _lbl_queue.set_text(f"Epsilon: {agent.epsilon:.4f}")
    for lbl in (_lbl_plan_state, _lbl_exp_state, _lbl_expanding, _lbl_path,
                _lbl_res_pos, _lbl_res_hits, _lbl_res_flags, _lbl_actions,
                _lbl_tele_origin, _lbl_tele_tried, _lbl_tele_exit, _lbl_tele_step,
                _lbl_fire_ctr, _lbl_fire_dir, _lbl_fire_flags,
                _lbl_rec_path, _lbl_rec_dir, _lbl_rec_flags):
        lbl.set_text("")


def draw_maze(maze: list[list[MazeCell]], fire_cells: set[tuple[int, int]]):
    walls: list[tuple[int, int, int, int]] = []
    for rowNum, row in enumerate(maze):
        for colNum, cell in enumerate(row):
            cell_x, cell_y = convert_grid_point_to_image_point((colNum, rowNum))
            cell_x -= camera_x
            cell_y -= camera_y
            if cell.top_square is None:
                walls.append((cell_x, cell_y, cell_x + 16, cell_y))
            if cell.left_square is None:
                walls.append((cell_x, cell_y, cell_x, cell_y + 16))
            if cell.right_square is None:
                walls.append((cell_x + 16, cell_y, cell_x + 16, cell_y + 16))
            if cell.bottom_square is None:
                walls.append((cell_x, cell_y + 16, cell_x + 16, cell_y + 16))
            if cell.state == CellState.START:
                pygame.draw.rect(screen, START_COLOR, pygame.Rect(cell_x, cell_y, 16, 16))
            elif cell.state == CellState.GOAL:
                pygame.draw.rect(screen, GOAL_COLOR, pygame.Rect(cell_x, cell_y, 16, 16))
            elif (colNum, rowNum) in fire_cells:
                screen.blit(FLAME_HAZARD, (cell_x, cell_y))
            elif cell.state == CellState.CONFUSION:
                pygame.draw.rect(screen, CONFUSION_COLOR, pygame.Rect(cell_x, cell_y, 16, 16))
            elif cell.state == CellState.GREEN_TELEPORT:
                screen.blit(GREEN_TELEPORT, (cell_x, cell_y))
            elif cell.state == CellState.PURPLE_TELEPORT:
                screen.blit(PURPLE_TELEPORT, (cell_x, cell_y))
            elif cell.state == CellState.YELLOW_TELEPORT:
                screen.blit(YELLOW_TELEPORT, (cell_x, cell_y))
    for wall in walls:
        pygame.draw.line(screen, WALL_COLOR, (wall[0], wall[1]), (wall[2], wall[3]), 2)


def color_cells(cells, color):
    for grid_point in cells:
        world_x, world_y = convert_grid_point_to_image_point(grid_point)
        screen_x = world_x - camera_x
        screen_y = world_y - camera_y
        pygame.draw.rect(screen, color, pygame.Rect(screen_x, screen_y, 16, 16))


def draw_little_guy(position: tuple[int, int]):
    world_x, world_y = convert_grid_point_to_image_point(position)
    screen.blit(LITTLE_GUY, (world_x - camera_x, world_y - camera_y))


def start_new_episode(env: AStarMazeEnvironment, agent: AStarAgent, episodes: list) -> tuple[TurnResult, int]:
    episodes.append(env.get_episode_stats())
    start_pos = env.reset()
    agent.reset_episode()
    result = TurnResult()
    result.current_position = start_pos
    return result, 0


_GEN_FIRE_ADVANCE = pygame.USEREVENT + 2  # timer event to advance fire rotation


def _run_generated_viewer(seed: int, size: int) -> None:
    global camera_x, camera_y
    from maze_generator import generate_maze, GridMazeEnvironment, _trace_path
    print(f"Generating maze seed={seed} size={size} ...")
    graph = generate_maze(seed=seed, size=size)
    env = GridMazeEnvironment(graph)
    env.reset()

    solution_path = {
        cell.position
        for cell in _trace_path(env._start_cell, env._goal_cell)
    }
    print(f"  {size}×{size} cells  |  fire hazards: {len(env._fire_centers)}  |  path length: {len(solution_path)}")
    print(f"  WASD to scroll, Q/Esc to quit, fire rotates every 500 ms")

    pygame.time.set_timer(_GEN_FIRE_ADVANCE, 500)
    camera_x = 0
    camera_y = 0

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
            if event.type == _GEN_FIRE_ADVANCE:
                env._fire_turn += 1

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]: camera_y -= CAM_SPEED_PX_PER_FRAME
        if keys[pygame.K_s]: camera_y += CAM_SPEED_PX_PER_FRAME
        if keys[pygame.K_a]: camera_x -= CAM_SPEED_PX_PER_FRAME
        if keys[pygame.K_d]: camera_x += CAM_SPEED_PX_PER_FRAME

        screen.fill(BACKGROUND_COLOR)
        color_cells(solution_path, TRAVERSING_COLOR)
        draw_maze(env._graph, env._active_fire_cells())
        pygame.display.set_caption(
            f"Generated Maze  seed={seed}  fire_rot={env._fire_turn % 4}"
        )
        pygame.display.flip()
        clock.tick(FRAMES_PER_SECOND)

    pygame.time.set_timer(_GEN_FIRE_ADVANCE, 0)
    pygame.quit()


def _run_rl_viewer(load_path: str, use_generated: bool, seed: int, size: int) -> None:
    global camera_x, camera_y
    from dqn_agent import DQNAgent

    agent = DQNAgent()
    agent.load(load_path)
    agent.epsilon = 0.0  # greedy

    if use_generated:
        from maze_generator import generate_maze, GridMazeEnvironment
        graph = generate_maze(seed=seed, size=size, num_fire_hazards=0, num_confusion=0)
        env = GridMazeEnvironment(graph)
        print(f"RL viewer: generated maze seed={seed} size={size}")
    else:
        env = AStarMazeEnvironment('training')
        print("RL viewer: training maze")

    print("  P pause/unpause  |  T step×1  |  Y step×5  |  WASD scroll  |  Q/Esc quit")

    start_pos = env.reset()
    agent._start_pos = start_pos
    agent._fire_turn = env._fire_turn
    agent.reset_episode()

    current_position = start_pos
    rl_result = None   # None on first step so plan_turn initialises correctly
    step_count = 0
    episode_count = 0
    paused = True

    def do_step() -> None:
        nonlocal rl_result, current_position, step_count, episode_count, start_pos
        actions = agent.plan_turn(rl_result)
        rl_result = env.step(actions)
        current_position = rl_result.current_position
        step_count += 1
        if rl_result.is_goal_reached or step_count >= MAX_STEPS_PER_EPISODE:
            episode_count += 1
            start_pos = env.reset()
            agent._start_pos = start_pos
            agent._fire_turn = env._fire_turn
            agent.reset_episode()
            current_position = start_pos
            rl_result = None
            step_count = 0

    pygame.time.set_timer(TICK_EVENT, 1000 // TICKS_PER_SECOND)
    clock = pygame.time.Clock()
    running = True

    while running:
        time_delta = clock.tick(FRAMES_PER_SECOND) / 1000.0

        for event in pygame.event.get():
            manager.process_events(event)
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                if event.key == pygame.K_p:
                    paused = not paused
                if event.key == pygame.K_t and paused:
                    do_step()
                if event.key == pygame.K_y and paused:
                    for _ in range(5):
                        do_step()
            if event.type == TICK_EVENT:
                for _ in range(STEPS_PER_TICK):
                    if not paused:
                        do_step()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]: camera_y -= CAM_SPEED_PX_PER_FRAME
        if keys[pygame.K_s]: camera_y += CAM_SPEED_PX_PER_FRAME
        if keys[pygame.K_a]: camera_x -= CAM_SPEED_PX_PER_FRAME
        if keys[pygame.K_d]: camera_x += CAM_SPEED_PX_PER_FRAME

        screen.fill(BACKGROUND_COLOR)
        color_cells(agent._episode_visited, VISITED_COLOR)
        draw_little_guy(current_position)
        draw_maze(env._graph, env._active_fire_cells())
        draw_rl_ui(agent, paused, episode_count, step_count)
        manager.update(time_delta)
        manager.draw_ui(screen)
        pygame.display.flip()

    pygame.time.set_timer(TICK_EVENT, 0)
    pygame.quit()


if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--generated', action='store_true',
                         help='Visualise a procedurally generated maze instead of the real one')
    _parser.add_argument('--seed', type=int, default=42,
                         help='RNG seed for generated maze (default: 42)')
    _parser.add_argument('--size', type=int, default=64,
                         help='Logical grid size for generated maze (default: 64)')
    _parser.add_argument('--rl', action='store_true',
                         help='Run the DQN agent instead of A*')
    _parser.add_argument('--load', type=str, default='model.pt',
                         help='Model weights to load for --rl (default: model.pt)')
    _args, _ = _parser.parse_known_args()

    if _args.rl:
        _run_rl_viewer(_args.load, _args.generated, _args.seed, _args.size)
        sys.exit(0)

    if _args.generated:
        _run_generated_viewer(_args.seed, _args.size)
        sys.exit(0)

    hazard_testing: bool = True
    agent: Agent = AStarAgent()
    mmAgent: ManualMovementAgent | None = agent if hazard_testing else None
    env = AStarMazeEnvironment("manual_hazards" if hazard_testing else "training")

    start_pos = env.reset()
    last_result = TurnResult()
    last_result.current_position = start_pos

    step_count = 0
    episode_count = 0
    episodes: list[dict] = []
    last_actions: list[Action] = []

    pygame.time.set_timer(TICK_EVENT, 1000 // TICKS_PER_SECOND)
    running = True
    clock = pygame.time.Clock()
    paused = True

    while running:
        time_delta = clock.tick(FRAMES_PER_SECOND) / 1000.0

        for event in pygame.event.get():
            manager.process_events(event)

            if event.type == pygame.QUIT:
                running = False

            if event.type == TICK_EVENT:
                for _ in range(STEPS_PER_TICK):
                    if not paused:
                        last_actions = agent.plan_turn(last_result)
                        last_result = env.step(last_actions)
                        step_count += 1
                        if last_result.is_goal_reached or step_count >= MAX_STEPS_PER_EPISODE:
                            last_result, step_count = start_new_episode(env, agent, episodes)
                            episode_count += 1

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    paused = not paused
                if event.key == pygame.K_t and paused:
                    last_actions = agent.plan_turn(last_result)
                    last_result = env.step(last_actions)
                    step_count += 1
                    if last_result.is_goal_reached or step_count >= MAX_STEPS_PER_EPISODE:
                        last_result, step_count = start_new_episode(env, agent, episodes)
                        episode_count += 1
                if event.key == pygame.K_y and paused:
                    for i in range(5):
                        last_actions = agent.plan_turn(last_result)
                        last_result = env.step(last_actions)
                        step_count += 1

                if hazard_testing:
                    assert(mmAgent is not None)
                    if event.key == pygame.K_RIGHT:
                        mmAgent.set_action(Action.MOVE_RIGHT)
                    elif event.key == pygame.K_LEFT:
                        mmAgent.set_action(Action.MOVE_LEFT)
                    elif event.key == pygame.K_UP:
                        mmAgent.set_action(Action.MOVE_UP)
                    elif event.key == pygame.K_DOWN:
                        mmAgent.set_action(Action.MOVE_DOWN)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            camera_y -= CAM_SPEED_PX_PER_FRAME
        if keys[pygame.K_s]:
            camera_y += CAM_SPEED_PX_PER_FRAME
        if keys[pygame.K_a]:
            camera_x -= CAM_SPEED_PX_PER_FRAME
        if keys[pygame.K_d]:
            camera_x += CAM_SPEED_PX_PER_FRAME

        screen.fill(BACKGROUND_COLOR)

        expanded = [pos for pos, cell in agent.memory.items() if cell.fully_expanded]
        color_cells(expanded, VISITED_COLOR)
        color_cells([cell.pos for cell in agent.open_queue], OPEN_QUEUE_COLOR)
        color_cells(agent.current_path, TRAVERSING_COLOR)
        color_cells(agent._recovery_path, TRAVERSING_COLOR)
        if agent.expanding_cell is not None:
            color_cells([agent.expanding_cell.pos], EXPANDING_COLOR)

        draw_little_guy(last_result.current_position)
        draw_maze(env._graph, env._active_fire_cells())
        draw_ui(agent, last_result, last_actions, paused, episode_count, step_count)

        manager.update(time_delta)
        manager.draw_ui(screen)
        pygame.display.flip()

    pygame.quit()
