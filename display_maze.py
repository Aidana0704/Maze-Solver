from pygame import Color
import pygame
from load_image import MazeCell, convert_grid_point_to_image_point, CellState
from maze_solver import AStarAgent, AStarMazeEnvironment, TurnResult
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

ui_frame = UIPanel(relative_rect=pygame.Rect(575, 11, 218, 218), manager=manager)

paused_text = UILabel(relative_rect=pygame.Rect(10, 10, 200, 20),
                      text="Paused?: ",
                      manager=manager, container=ui_frame)

episode_text = UILabel(relative_rect=pygame.Rect(10, 30, 200, 20),
                       text="Episode: ",
                       manager=manager, container=ui_frame)

steps_text = UILabel(relative_rect=pygame.Rect(10, 50, 200, 20),
                     text="Steps: ",
                     manager=manager, container=ui_frame)

explored_text = UILabel(relative_rect=pygame.Rect(10, 70, 200, 20),
                        text="Cells explored: ",
                        manager=manager, container=ui_frame)

expansion_state_text = UILabel(relative_rect=pygame.Rect(10, 90, 200, 20),
                               text="Expansion state: ",
                               manager=manager, container=ui_frame)


def draw_ui(agent: AStarAgent, paused: bool, episode_count: int, step_count: int):
    paused_text.set_text(f"Paused?: {'true' if paused else 'false'}")
    episode_text.set_text(f"Episode: {episode_count}")
    steps_text.set_text(f"Steps: {step_count} / {MAX_STEPS_PER_EPISODE}")
    explored_text.set_text(f"Cells explored: {len(agent.memory)}")
    expansion_state_text.set_text(f"Expansion: {agent.expansion_state.name}")


def draw_maze(maze: list[list[MazeCell]]):
    border = [(1, 0), (1, 1025), (1025, 1025), (1025, 0)]
    walls: list[tuple[int, int, int, int]] = []
    pygame.draw.lines(screen, WALL_COLOR, True, border, 2)
    for rowNum, row in enumerate(maze):
        for colNum, cell in enumerate(row):
            cell_x, cell_y = convert_grid_point_to_image_point((colNum, rowNum))
            cell_x -= camera_x
            cell_y -= camera_y
            if cell.right_square is None:
                walls.append((cell_x + 16, cell_y, cell_x + 16, cell_y + 16))
            if cell.bottom_square is None:
                walls.append((cell_x, cell_y + 16, cell_x + 16, cell_y + 16))
            if cell.state == CellState.START:
                pygame.draw.rect(screen, START_COLOR, pygame.Rect(cell_x, cell_y, 16, 16))
            elif cell.state == CellState.GOAL:
                pygame.draw.rect(screen, GOAL_COLOR, pygame.Rect(cell_x, cell_y, 16, 16))
            elif cell.state == CellState.HAZARD:
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


if __name__ == "__main__":
    agent = AStarAgent()
    env = AStarMazeEnvironment('training')

    start_pos = env.reset()
    last_result = TurnResult()
    last_result.current_position = start_pos

    step_count = 0
    episode_count = 0
    episodes: list[dict] = []

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
                        actions = agent.plan_turn(last_result)
                        last_result = env.step(actions)
                        step_count += 1
                        if last_result.is_goal_reached or step_count >= MAX_STEPS_PER_EPISODE:
                            last_result, step_count = start_new_episode(env, agent, episodes)
                            episode_count += 1

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    paused = not paused
                if event.key == pygame.K_t and paused:
                    actions = agent.plan_turn(last_result)
                    last_result = env.step(actions)
                    step_count += 1
                    if last_result.is_goal_reached or step_count >= MAX_STEPS_PER_EPISODE:
                        last_result, step_count = start_new_episode(env, agent, episodes)
                        episode_count += 1

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
        draw_little_guy(last_result.current_position)
        draw_maze(env._graph)
        draw_ui(agent, paused, episode_count, step_count)
        manager.update(time_delta)
        manager.draw_ui(screen)
        pygame.display.flip()

    pygame.quit()
