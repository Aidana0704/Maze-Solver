from pygame import Color
import pygame
from load_image import MazeCell, convert_grid_point_to_image_point, load_image_into_graph, CellState
from solve_maze import AStarSolver, manhattan_distance
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
PURPLE_TELEPORT = pygame.image.load("purple_teleport.png")
GREEN_TELEPORT = pygame.image.load("green_teleport.png")
YELLOW_TELEPORT = pygame.image.load("yellow_teleport.png")

camera_x: int = 0
camera_y: int = 0

pygame.init()

screen = pygame.display.set_mode((800, 800))
pygame.display.set_caption("Maze Solver")

manager = pygame_gui.UIManager((800, 800), 'theme.json')

running = True


ui_frame = UIPanel(relative_rect=pygame.Rect(575, 11, 218, 218),
                        manager=manager)

paused_text = UILabel(relative_rect=pygame.Rect(10, 10, 200, 20),
                        text="Paused?: ",
                        manager=manager,
                        container=ui_frame)

turn_count_text = UILabel(relative_rect=pygame.Rect(10, 30, 200, 20),
                            text="# of Turns: ",
                            manager=manager,
                            container=ui_frame)

manhattan_distance_text = UILabel(relative_rect=pygame.Rect(10, 50, 200, 20),
                                    text="Manhattan Distance: ",
                                    manager=manager,
                                    container=ui_frame)

cost_text = UILabel(relative_rect=pygame.Rect(10, 70, 200, 20),
                    text="Cost: ",
                    manager=manager,
                    container=ui_frame)
    
def draw_ui(solver: AStarSolver, paused: bool):
    paused_text.set_text(f"Paused?: {"true" if paused else "false"}")
    turn_count_text.set_text(f"# of Turns: {solver.turn_count}")
    manhattan_distance_text.set_text(f"Manhattan Distance: {manhattan_distance(solver.character_position, solver.goal_pos)}")
    cost_text.set_text(f"Cost: {solver.current_node.cost}")



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

    paused = True

    while running:
        time_delta = clock.tick(FRAMES_PER_SECOND) // 1000.0

        for event in pygame.event.get():
            manager.process_events(event)
            
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
                elif event.key == pygame.K_RIGHT:
                    solver.move_right()
                elif event.key == pygame.K_UP:
                    solver.move_up()
                elif event.key == pygame.K_DOWN:
                    solver.move_down()
                elif event.key == pygame.K_LEFT:
                    solver.move_left()
                
            
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

        # draw_path(solution_path)
        color_cells(solver.visited, VISITED_COLOR)
        color_cells(solver.character_path, TRAVERSING_COLOR)
        color_cells(solver.ans, PATH_COLOR)
        draw_little_guy(solver.character_position)
        
        draw_maze(maze_graph)
        draw_ui(solver, paused)
        manager.update(time_delta)
        manager.draw_ui(screen)
        pygame.display.flip()

    pygame.quit()