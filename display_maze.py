from pygame import Color
import pygame
from load_image import MazeCell, convert_grid_point_to_image_point, load_image_into_graph, CellState
from solve_maze import get_start_and_goal_pos, solve_maze_astar, AStarSolver

BACKGROUND_COLOR = Color(255, 255, 255)
WALL_COLOR = Color(0, 0, 0)
START_COLOR = Color(0, 0, 255)
GOAL_COLOR = Color(0, 255, 0)
PATH_COLOR = Color(255, 0, 0)
VISITED_COLOR = Color(25, 216, 255)
TRAVERSING_COLOR = Color(255, 157, 0)
HAZARD_COLOR = Color(255, 216, 0)
CONFUSION_COLOR = Color(188, 37, 168)

CAM_SPEED_PX_PER_SECOND = 300
FRAMES_PER_SECOND = 60

TICKS_PER_SECOND = 1000
STEPS_PER_TICK = 30
TICK_EVENT = pygame.USEREVENT + 1

CAM_SPEED_PX_PER_FRAME = CAM_SPEED_PX_PER_SECOND // FRAMES_PER_SECOND

little_guy_row: int = 0
little_guy_col: int = 0

LITTLE_GUY = pygame.image.load("lil_guy.png")
FLAME_HAZARD = pygame.image.load("flame_hazard.png")

camera_x: int = 0
camera_y: int = 0

pygame.init()

screen = pygame.display.set_mode((800, 800))
pygame.display.set_caption("Maze Solver")

running = True

def draw_maze(maze: list[list[MazeCell]]):
    border = [(1, 0), (1, 1025), (1025, 1025), (1025, 0)]
    walls: list[tuple[int, int, int, int]] = []
    pygame.draw.lines(screen, WALL_COLOR, True, border, 2)
    for rowNum, row in enumerate(maze):
        for colNum, cell in enumerate(row):
            cell_x, cell_y = convert_grid_point_to_image_point((colNum, rowNum))
            cell_x -= camera_x
            cell_y -= camera_y
            if (cell.right_square == None):
                walls.append((cell_x + 16, cell_y, cell_x + 16, cell_y + 16))
            
            if (cell.bottom_square == None):
                walls.append((cell_x, cell_y + 16, cell_x + 16, cell_y + 16))
            
            if cell.state == CellState.START:
                pygame.draw.rect(screen, START_COLOR, pygame.Rect(cell_x, cell_y, 16, 16))
            elif cell.state == CellState.GOAL:
                pygame.draw.rect(screen, GOAL_COLOR, pygame.Rect(cell_x, cell_y, 16, 16))
            elif cell.state == CellState.HAZARD:
                screen.blit(FLAME_HAZARD, (cell_x, cell_y))
            elif cell.state == CellState.CONFUSION:
                pygame.draw.rect(screen, CONFUSION_COLOR, pygame.Rect(cell_x, cell_y, 16, 16))
    
    for wall in walls:
        pygame.draw.line(screen, WALL_COLOR, (wall[0], wall[1]), (wall[2], wall[3]), 2)

    pygame.display.flip()

def color_cells(cells, color):
    for grid_point in cells:
        world_x, world_y = convert_grid_point_to_image_point(grid_point)
        screen_x, screen_y = (world_x - camera_x, world_y - camera_y)

        pygame.draw.rect(screen, color, pygame.Rect(screen_x, screen_y, 16, 16))

def draw_little_guy(grid_point):
    world_x, world_y = convert_grid_point_to_image_point(grid_point)
    screen_x, screen_y = (world_x - camera_x, world_y - camera_y)
    screen.blit(LITTLE_GUY, (screen_x, screen_y))

if __name__ == "__main__":
    maze_graph = load_image_into_graph("MAZE_1.png")
    # below lines are for static path generation, commented out in favor of procedural visual solution
    # start, goal = get_start_and_goal_pos(maze_graph)
    # solution_path = solve_maze_astar(maze_graph, start, goal)

    # below lines are for procedural visual solution
    solver = AStarSolver(maze_graph)
    pygame.time.set_timer(TICK_EVENT, 1000 // TICKS_PER_SECOND)

    running = True
    clock = pygame.time.Clock()

    paused = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == TICK_EVENT:
                for _ in range(STEPS_PER_TICK):
                    if not paused:
                        solver.step()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    paused = not paused
                
                if event.key == pygame.K_t and paused:
                    solver.step()
            
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            camera_y -= CAM_SPEED_PX_PER_FRAME
        if keys[pygame.K_s]:
            camera_y += CAM_SPEED_PX_PER_FRAME
        if keys[pygame.K_a]:
            camera_x -= CAM_SPEED_PX_PER_FRAME
        if keys[pygame.K_d]:
            camera_x += CAM_SPEED_PX_PER_FRAME

        clock.tick(FRAMES_PER_SECOND)
        screen.fill(BACKGROUND_COLOR)

        # draw_path(solution_path)
        color_cells(solver.visited, VISITED_COLOR)
        color_cells(solver.character_path, TRAVERSING_COLOR)
        color_cells(solver.ans, PATH_COLOR)
        draw_little_guy(solver.character_position)
        
        draw_maze(maze_graph)
    
    pygame.quit()