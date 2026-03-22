from __future__ import annotations
from load_image import CellState, MazeCell
from heapq import heappop, heappush


def manhattan_distance(cell_coords: tuple[int, int], goal_coords: tuple[int, int]):
    return abs(goal_coords[0] - cell_coords[1]) + abs(goal_coords[1] - cell_coords[0])

class CellAStarInfo:
    goal_coords: tuple[int, int] = (0, 0)
    def __init__(self, cell: MazeCell, cost: int, coords: tuple[int, int], parent: CellAStarInfo | None = None):
        self.parent: None | CellAStarInfo = parent
        self.cell: MazeCell = cell
        self.cost: int = cost
        self.coords: tuple[int, int] = coords
    
    def f(self):
        return self.cost + manhattan_distance(self.coords, CellAStarInfo.goal_coords)

    def __lt__(self, other: CellAStarInfo):
        return self.f() < other.f()
    

# start pos and goal pos to be interpreted as (row, column) and everything else is to be interpreted as (x, y)
# y decreases upwards, x increases rightwards
# don't ask me why i did it this way. i'm tired chief. i'm tired.

# (it's because that's how pygame works)
# i'm a fucking chud bro

def get_start_and_goal_pos(maze: list[list[MazeCell]]) -> tuple[tuple[int, int], tuple[int, int]]:
    goal = None
    start = None
    for rowNum, row in enumerate(maze):
        for colNum, cell in enumerate(row):
            if cell.state == CellState.GOAL:
                goal = (rowNum, colNum)
            elif cell.state == CellState.START:
                start = (rowNum, colNum)
            
            if (goal is not None and start is not None):
                return (start, goal)
    

    return (start, goal)

# THIS IS AN ALGORITHM PURELY USED FOR THE VISUAL EXPLORATION OF THE MAZE.
# IT HAS NOTHING TO DO WITH THE A* BENEATH.

def get_path_between_expanded_nodes(current_node: CellAStarInfo, next_node: CellAStarInfo):
    next_node_ancestors = set()
    path_second_half = []
    curr = next_node
    while curr is not None:
        next_node_ancestors.add(curr.coords)
        path_second_half.append(curr.coords)
        curr = curr.parent

    path_second_half.reverse()

    curr = current_node

    path_first_half = []

    while curr.coords not in next_node_ancestors:
        path_first_half.append(curr.coords)
        curr = curr.parent

    lca_index = path_second_half.index(curr.coords)

    return path_first_half + path_second_half[lca_index:]

def solve_maze_astar(maze: list[list[MazeCell]], start_pos: tuple[int, int], goal_pos: tuple[int, int]) -> list[tuple[int]]:
    CellAStarInfo.goal_coords = goal_pos
    visited = set()
    q = [CellAStarInfo(maze[start_pos[0]][start_pos[1]], 0, (start_pos[1], start_pos[0]))]
    while (len(q) > 0):
        current_node = q[0]
        heappop(q)
        if (current_node.coords in visited):
            continue

        visited.add(current_node.coords)

        current_node_cell = current_node.cell

        if current_node.coords == (goal_pos[1], goal_pos[0]):
            ans = []
            c = current_node
            while c is not None:
                ans.append(c.coords)
                c = c.parent
            
            return ans
            

        if current_node_cell.bottom_square is not None:
            heappush(q, CellAStarInfo(current_node_cell.bottom_square, current_node.cost + 1, current_node_cell.bottom_square.position, current_node))
        if current_node_cell.right_square is not None:
            heappush(q, CellAStarInfo(current_node_cell.right_square, current_node.cost + 1, current_node_cell.right_square.position, current_node))
        if current_node_cell.left_square is not None:
            heappush(q, CellAStarInfo(current_node_cell.left_square, current_node.cost + 1, current_node_cell.left_square.position, current_node))
        if current_node_cell.top_square is not None:
            heappush(q, CellAStarInfo(current_node_cell.top_square, current_node.cost + 1, current_node_cell.top_square.position, current_node))

    return [] # no path was found

# class used for step by step solving of the problem
# again, initial and end position is expected to be (row, column) when passed
# to constructor, but is automatically converted during the constructor

# i loooovve being inconsistent

class AStarSolver:
    def __init__(self, graph: list[list[MazeCell]]):
        self.graph = graph
        start, goal = get_start_and_goal_pos(self.graph)

        self.turn_count = 0
        self.start_pos = (start[1], start[0])
        self.goal_pos = (goal[1], goal[0])
        self.iterations: int = 0
        self.character_position = self.start_pos
        self.current_pos = self.start_pos
        self.visited: set[tuple[int, int]] = set()
        CellAStarInfo.goal_coords = self.goal_pos
        self.queue: list[CellAStarInfo] = [CellAStarInfo(graph[start[0]][start[1]], 0, self.start_pos)]
        self.start_node = self.queue[0]
        self.finished: bool = False
        self.started: bool = False
        self.ans: None | list[tuple[int, int]] = []
        self.character_path: list[tuple[int, int]] = []
        self.current_node = self.queue[0]
        self.confusionSteps: int = 0

    def _move_to(self, cell: MazeCell):
        if self.started:
            return
        
        if (cell.state == CellState.HAZARD):
            self.current_node = self.start_node
            self.character_position = self.current_node.coords
            self.current_pos = self.character_position
            return

        if (cell.state == CellState.CONFUSION):
            self.confusionSteps = 5
        
        self.current_node = CellAStarInfo(cell, 0, cell.position)
        self.character_position = cell.position
        self.current_pos = cell.position


    _OPPOSITES = {
        'left_square':   'right_square',
        'right_square':  'left_square',
        'top_square':    'bottom_square',
        'bottom_square': 'top_square',
    }

    def _move(self, direction: str):
        if self.confusionSteps > 0:
            self.confusionSteps -= 1
            direction = self._OPPOSITES[direction]
        cell = getattr(self.current_node.cell, direction)
        if cell is not None:
            self._move_to(cell)

    def move_left(self):  self._move('left_square')
    def move_right(self): self._move('right_square')
    def move_up(self):    self._move('top_square')
    def move_down(self):  self._move('bottom_square')

    def step(self):
        if (self.finished):
            return

        if (len(self.character_path) != 0):
            self.turn_count += 1
            self.character_position = self.character_path.pop(0)
            return

        self.started = True

        current_node = self.queue[0]
        heappop(self.queue)
        if (current_node.coords in self.visited):
            return


        self.character_path = get_path_between_expanded_nodes(self.current_node, current_node)
        self.current_node = current_node
        self.current_pos = current_node.coords


        self.visited.add(current_node.coords)

        current_node_cell = current_node.cell

        if current_node.coords == self.goal_pos:
            ans = []
            c = current_node
            while c is not None:
                ans.append(c.coords)
                c = c.parent

            self.ans = ans
            self.finished = True
            self.character_path = []
            self.character_position = self.start_pos
            return


        if current_node_cell.bottom_square is not None:
            heappush(self.queue, CellAStarInfo(current_node_cell.bottom_square, current_node.cost + 1, current_node_cell.bottom_square.position, current_node))
        if current_node_cell.right_square is not None:
            heappush(self.queue, CellAStarInfo(current_node_cell.right_square, current_node.cost + 1, current_node_cell.right_square.position, current_node))
        if current_node_cell.left_square is not None:
            heappush(self.queue, CellAStarInfo(current_node_cell.left_square, current_node.cost + 1, current_node_cell.left_square.position, current_node))
        if current_node_cell.top_square is not None:
            heappush(self.queue, CellAStarInfo(current_node_cell.top_square, current_node.cost + 1, current_node_cell.top_square.position, current_node))