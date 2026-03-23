from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Optional
from PIL import Image
from PIL.ImageFile import ImageFile
import numpy as np
from numpy import array # only used for pretty-printing the matrix :)
import sys

HAZARD_COLOR = (247, 118, 55)
CONFUSION_COLOR = (156, 117, 60)
CONFUSION_COLOR_2 = (199, 154, 65)
YELLOW_TELEPORT_COLOR = (255, 111, 56)
YELLOW_TELEPORT_COLOR_2 = (240, 192, 76)
PURPLE_TELEPORT_COLOR = (135, 105, 193)
PURPLE_TELEPORT_COLOR_2 = (143, 101, 227)
GREEN_TELEPORT_COLOR = (91, 220, 148)
GREEN_TELEPORT_COLOR_2 = (65, 158, 112)

class CellState(Enum):
    EMPTY = 0
    START = 1
    GOAL = 2
    PIT = 3
    PURPLE_TELEPORT = 4
    CONFUSION = 5
    HAZARD = 6
    GREEN_TELEPORT = 7
    YELLOW_TELEPORT = 8
    def __repr__(self):
        return self.name[0]


class SampleResult:
    wall_below: bool
    wall_right: bool
    state: CellState = CellState.EMPTY

class MazeCell:
    right_square: Optional[MazeCell] = None
    left_square: Optional[MazeCell] = None
    top_square: Optional[MazeCell] = None
    bottom_square: Optional[MazeCell] = None
    state: CellState = CellState.EMPTY
    position: tuple[int, int] = (0, 0)  # (x, y) i.e. (col, row)

    def __repr__(self) -> str:
        output = ""
        letter = ('E', 'E̲')
        if (self.state == CellState.GOAL):
            letter = ('G', 'G̲')
        elif (self.state == CellState.START):
            letter = ('S', 'S̲')
        if self.left_square is None:
            output += "|"
        else:
            output += " "
        if self.bottom_square is None:
            output += letter[1]
        else:
            output += letter[0]
        if self.right_square is None:
            output += "|"
        else:
            output += " "
        return output
            

def color_distance(c1, c2):
    return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2) ** 0.5

def convert_grid_point_to_image_point(grid_point: tuple[int, int]) -> tuple[int, int]:
    """Returns the top left position in image space of the given grid space coordinate."""
    image_x = grid_point[0] * 16 + 1
    image_y = grid_point[1] * 16 + 1
    return (image_x, image_y)

def sample_point(pos: tuple[int, int], image: ImageFile) -> SampleResult:
    #print(f"sampling gridpoint {pos}")
    sample_result = SampleResult()
    image_point = convert_grid_point_to_image_point(pos)

    #print(f"sampling point {image_point}")

    right_wall_pos = (image_point[0] + 15, image_point[1] + 7)
    bottom_wall_pos = (image_point[0] + 7, image_point[1] + 15)
    center_pos = (image_point[0] + 7, image_point[1] + 7)
    teleporter_color_pos = (image_point[0], image_point[1] + 7)

    # if we sample black, that means there's a wall.
    # except NOT because for SOME REASON, THE EMOJIS ON THE MAZE
    # CAN AFFECT THE COLOR OF THE WALLS???
    # ????
    # WHO MADE THESE IMAGES???
    # ts pmo sm icl 💔

    print(image.getpixel(bottom_wall_pos))
    bottom_r, bottom_g, bottom_b = image.getpixel(bottom_wall_pos)
    sample_result.wall_below = max(bottom_r, bottom_g, bottom_b) < 255//4
    right_r, right_g, right_b = image.getpixel(right_wall_pos)
    sample_result.wall_right = max(right_r, right_g, right_b) < 255//4

    middle_color = image.getpixel(center_pos)
    teleport_color = image.getpixel(teleporter_color_pos)
    
    if color_distance(middle_color[:3], HAZARD_COLOR) < 50:
        sample_result.state = CellState.HAZARD
    elif color_distance(middle_color[:3], CONFUSION_COLOR) < 50:
        sample_result.state = CellState.CONFUSION
    elif color_distance(middle_color[:3], CONFUSION_COLOR_2) < 50:
        sample_result.state = CellState.CONFUSION
    elif color_distance(teleport_color[:3], YELLOW_TELEPORT_COLOR) < 50:
        sample_result.state = CellState.YELLOW_TELEPORT
    elif color_distance(middle_color[:3], YELLOW_TELEPORT_COLOR_2) < 50:
        sample_result.state = CellState.YELLOW_TELEPORT
    elif color_distance(middle_color[:3], PURPLE_TELEPORT_COLOR) < 50:
        sample_result.state = CellState.PURPLE_TELEPORT
    elif color_distance(middle_color[:3], PURPLE_TELEPORT_COLOR_2) < 50:
        sample_result.state = CellState.PURPLE_TELEPORT
    elif color_distance(middle_color[:3], GREEN_TELEPORT_COLOR) < 50:
        sample_result.state = CellState.GREEN_TELEPORT
    elif color_distance(teleport_color[:3], GREEN_TELEPORT_COLOR_2) < 50:
        sample_result.state = CellState.GREEN_TELEPORT

    return sample_result

def load_image_into_graph(image_src: Path) -> list[list[MazeCell]]:
    loaded_map: list[list[MazeCell]] = [[MazeCell() for _ in range(64)] for _ in range(64)]

    for i, row in enumerate(loaded_map):
        for j, maze_cell in enumerate(row):
            maze_cell.position = (j, i)  # (x, y) = (col, row)
            if j > 0:
                maze_cell.left_square = row[j - 1]
            if j < len(row) - 1:
                maze_cell.right_square = row[j + 1]
            if i > 0:
                maze_cell.top_square = loaded_map[i - 1][j]
            if i < len(loaded_map) - 1:
                maze_cell.bottom_square = loaded_map[i + 1][j]

    with Image.open(image_src).convert("RGB") as img:
        # Sample image at specific discrete points
        row = 0
        while row <= 63:
            col = 0
            while col <= 63:
                sample_info = sample_point((col, row), img)
                loaded_map[row][col].state = sample_info.state
                if sample_info.wall_below and loaded_map[row][col].bottom_square is not None:
                    loaded_map[row][col].bottom_square.top_square = None
                    loaded_map[row][col].bottom_square = None
                if sample_info.wall_right and loaded_map[row][col].right_square is not None:
                    loaded_map[row][col].right_square.left_square = None
                    loaded_map[row][col].right_square = None
                
                col += 1
            row += 1
        
        # find exit
        for i in range(64):
            grid_spot = convert_grid_point_to_image_point((i, 0))
            grid_spot = (grid_spot[0] + 1, grid_spot[1])
            grid_r, grid_g, grid_b = img.getpixel(grid_spot)
            if max(grid_r, grid_g, grid_b) >= 255 * 4 // 5:
                print(f"found goal at {grid_spot}")
                loaded_map[0][i].state = CellState.GOAL
                break
        
        for i in range(64):
            grid_spot = convert_grid_point_to_image_point((i, 63))
            grid_spot = (grid_spot[0] + 1, grid_spot[1] + 15)
            grid_r, grid_g, grid_b = img.getpixel(grid_spot)
            if max(grid_r, grid_g, grid_b) >= 255 * 4 // 5:
                print(f"found start at {grid_spot}")
                loaded_map[63][i].state = CellState.START
                break

    teleport_colors = [CellState.PURPLE_TELEPORT, CellState.GREEN_TELEPORT, CellState.YELLOW_TELEPORT]
    for color in teleport_colors:
        pair = [cell for row in loaded_map for cell in row if cell.state == color]
        if len(pair) == 0:
            continue

        a, b = pair[0], pair[1]
        for _, dst, neighbors in [(a, b, [(a.right_square, 'left_square'),
                                            (a.left_square,  'right_square'),
                                            (a.top_square,   'bottom_square'),
                                            (a.bottom_square,'top_square')]),
                                    (b, a, [(b.right_square, 'left_square'),
                                            (b.left_square,  'right_square'),
                                            (b.top_square,   'bottom_square'),
                                            (b.bottom_square,'top_square')])]:
            for neighbor, back_attr in neighbors:
                if neighbor is not None:
                    setattr(neighbor, back_attr, dst)

    return loaded_map

if __name__ == "__main__":
    loaded_map = load_image_into_graph(Path("MAZE_0.png"))
    np.set_printoptions(threshold=sys.maxsize, linewidth=1000)
    print(array(loaded_map))
    print(len(loaded_map))
    print(len(loaded_map[0]))