from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Optional
from PIL import Image
from PIL.ImageFile import ImageFile
import numpy as np
from numpy import array # only used for pretty-printing the matrix :)
import sys


class CellState(Enum):
    EMPTY = 0
    START = 1
    GOAL = 2
    PIT = 3
    TELEPORT = 4
    CONFUSION = 5
    def __repr__(self):
        return self.name[0]


class SampleResult:
    wall_below: bool
    wall_right: bool

class MazeCell:
    right_square: Optional[MazeCell] = None
    left_square: Optional[MazeCell] = None
    top_square: Optional[MazeCell] = None
    bottom_square: Optional[MazeCell] = None
    state: CellState = CellState.EMPTY

    def __repr__(self) -> str:
        output = ""
        print("str called")
        if self.left_square is None:
            output += "|"
        else:
            output += " "
        if self.bottom_square is None:
            output += "E̲"
        else:
            output += "E"
        if self.right_square is None:
            output += "|"
        else:
            output += " "
        return output
            



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

    # if we sample black, that means there's a wall.

    sample_result.wall_below = image.getpixel(bottom_wall_pos) == (0, 0, 0, 255)
    sample_result.wall_right = image.getpixel(right_wall_pos) == (0, 0, 0, 255)


    return sample_result

def load_image_into_graph(image_src: Path) -> list[list[MazeCell]]:
    loaded_map: list[list[MazeCell]] = [[MazeCell() for i in range(64)] for j in range(64)]

    for i, row in enumerate(loaded_map):
        for j, maze_cell in enumerate(row):
            if j > 0:
                maze_cell.left_square = row[j - 1]
            if j < len(row) - 1:
                maze_cell.right_square = row[j + 1]
            if i > 0:
                maze_cell.top_square = loaded_map[i - 1][j]
            if i < len(loaded_map) - 1:
                maze_cell.bottom_square = loaded_map[i + 1][j]

    with Image.open(image_src) as img:
        # Sample image at specific discrete points
        row = 0
        while row <= 63:
            col = 0
            while col <= 63:
                cell_wall_info = sample_point((col, row), img)
                if cell_wall_info.wall_below and loaded_map[row][col].bottom_square is not None:
                    loaded_map[row][col].bottom_square.top_square = None
                    loaded_map[row][col].bottom_square = None
                if cell_wall_info.wall_right and loaded_map[row][col].right_square is not None:
                    loaded_map[row][col].right_square.left_square = None
                    loaded_map[row][col].right_square = None
                
                col += 1
            row += 1

    return loaded_map

if __name__ == "__main__":
    loaded_map = load_image_into_graph(Path("MAZE_0.png"))
    np.set_printoptions(threshold=sys.maxsize, linewidth=1000)
    print(array(loaded_map))
    print(len(loaded_map))
    print(len(loaded_map[0]))