from __future__ import annotations
from typing import override, NamedTuple
from dataclasses import dataclass
from enum import Enum
from load_image import CellState, MazeCell, load_image_into_graph
from solve_maze import get_start_and_goal_pos
from heapq import heappop, heappush, heapify

class Action(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4


class TurnResult:
    def __init__(self):
        self.wall_hits: int = 0
        self.current_position: tuple[int, int] = (0, 0)
        self.is_dead: bool = False
        self.is_confused: bool = False
        self.is_goal_reached: bool = False
        self.teleported: bool = False
        self.actions_executed: int = 0


class MazeEnvironment:
    def __init__(self, maze_id: str):
        """
        Initialize maze environment

        Args:
            maze_id: 'training' or 'testing'
        """
        pass

    def reset(self) -> tuple[int, int]:
        """
        Reset environment for new episode

        Returns:
            Starting position coordinates
        """
        pass

    def step(self, actions: list[Action]) -> TurnResult:
        """
        Execute a turn with given actions

        Args:
            actions: List of 1-5 Action objects

        Returns:
            TurnResult with feedback

        Raises:
            ValueError: If actions list empty or >5 actions
        """
        pass

    def get_episode_stats(self) -> dict:
        """
        Get statistics for current episode

        Returns:
            Dictionary containing:
            - turns_taken: int
            - deaths: int
            - confused: int
            - cells_explored: int
            - goal_reached: bool
        """
        pass


_MAZE_FILES = {
    'training': 'MAZE_0.png',
    'manual_hazards': 'MAZE_1.png',
    'testing':  'MAZE_0.png',
}

_ACTION_TO_NEIGHBOR = {
    Action.MOVE_UP: 'top_square',
    Action.MOVE_DOWN: 'bottom_square',
    Action.MOVE_LEFT:  'left_square',
    Action.MOVE_RIGHT: 'right_square',
}

_CONFUSION_OPPOSITE = {
    Action.MOVE_UP: Action.MOVE_DOWN,
    Action.MOVE_DOWN: Action.MOVE_UP,
    Action.MOVE_LEFT: Action.MOVE_RIGHT,
    Action.MOVE_RIGHT: Action.MOVE_LEFT,
    Action.WAIT: Action.WAIT,
}

_TELEPORT_STATES = {CellState.PURPLE_TELEPORT, CellState.GREEN_TELEPORT, CellState.YELLOW_TELEPORT}


class AStarMazeEnvironment(MazeEnvironment):
    def __init__(self, maze_id: str):
        self._graph = load_image_into_graph(_MAZE_FILES[maze_id])
        start_rc, goal_rc = get_start_and_goal_pos(self._graph)
        self._start_cell: MazeCell = self._graph[start_rc[0]][start_rc[1]]
        self._goal_pos: tuple[int, int] = (goal_rc[1], goal_rc[0])  # (x, y)
        AStarAgentMemoryCell.goal_pos = CellPosition(*self._goal_pos)
        self._current_cell: MazeCell = self._start_cell
        self._confusion_steps: int = 0
        self._turns: int = 0
        self._deaths: int = 0
        self._confused: int = 0
        self._explored: set[tuple[int, int]] = set()
        self._goal_reached: bool = False

    def reset(self) -> tuple[int, int]:
        self._current_cell = self._start_cell
        self._confusion_steps = 0
        self._turns = 0
        self._deaths = 0
        self._confused = 0
        self._explored = {self._start_cell.position}
        self._goal_reached = False
        return self._start_cell.position

    def step(self, actions: list[Action]) -> TurnResult:
        if len(actions) == 0 or len(actions) > 5:
            raise ValueError(f"actions must have 1–5 entries, got {len(actions)}")

        self._confusion_steps = 0
        result = TurnResult()

        for action in actions:
            if action == Action.WAIT:
                result.actions_executed += 1
                continue

            if self._confusion_steps > 0:
                self._confusion_steps -= 1
                action = _CONFUSION_OPPOSITE[action]

            next_cell: MazeCell | None = getattr(self._current_cell, _ACTION_TO_NEIGHBOR[action])
            if next_cell is None:
                result.wall_hits += 1
                result.actions_executed += 1
                continue

            self._current_cell = next_cell
            self._explored.add(self._current_cell.position)
            result.actions_executed += 1

            if self._current_cell.state == CellState.HAZARD:
                result.is_dead = True
                self._deaths += 1
                self._current_cell = self._start_cell
                self._explored.add(self._start_cell.position)
                continue

            if self._current_cell.state == CellState.CONFUSION:
                result.is_confused = True
                self._confused += 1
                self._confusion_steps = 5

            if self._current_cell.state in _TELEPORT_STATES:
                result.teleported = True

            if self._current_cell.position == self._goal_pos:
                result.is_goal_reached = True
                self._goal_reached = True
                print("goal reached!!! yay!!!!")
                break

        result.current_position = self._current_cell.position
        self._turns += 1
        return result

    def get_episode_stats(self) -> dict:
        return {
            'turns_taken': self._turns,
            'deaths': self._deaths,
            'confused': self._confused,
            'cells_explored': len(self._explored),
            'goal_reached': self._goal_reached,
        }


class Agent:
    """
    Base class for student implementations
    Students must implement this interface
    """

    def __init__(self):
        """
        Initialize agent with empty memory
        """
        self.memory = {}  # Students can structure as needed

    def plan_turn(self, last_result: TurnResult) -> list[Action]:
        """
        Plan next set of actions based on last turn result

        Args:
            last_result: Feedback from previous turn
                (None on first turn of episode)

        Returns:
            List of 1-5 actions to execute
        """
        raise NotImplementedError("Students must implement this method")

    def reset_episode(self):
        """
        Called at start of new episode
        Students can reset episode-specific state
        Memory can be retained for learning
        """
        pass


class CellPosition(NamedTuple):
    x: int
    y: int

@dataclass
class CellNeighbors:
    up: CellPosition | None = None
    right: CellPosition | None = None
    down: CellPosition | None = None
    left: CellPosition | None = None

class AStarAgentExpansionState(Enum):
    NOT_EXPANDING = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

class AStarAgentPlanningState(Enum):
    EXPANDING = 0
    TRAVERSING = 1
    START = 2

class AStarAgentMemoryCell:
    goal_pos: CellPosition = (0, 0)

    def __init__(self, state: CellState, pos: CellPosition, previous: AStarAgentMemoryCell | None):
        self.state = state
        self.pos = pos
        self.previous = previous
        self.neighbors = CellNeighbors()
        self.fully_expanded: bool = False
        self.cost: int = previous.cost + 1 if previous is not None else 0
        self.manhattan_distance = abs(AStarAgentMemoryCell.goal_pos.x - self.pos.x) + abs(AStarAgentMemoryCell.goal_pos.y - self.pos.y)
    
    def __repr__(self):
        return f"({self.pos.x}, {self.pos.y})"

    def __lt__(self, other: AStarAgentMemoryCell):
        return self.cost + self.manhattan_distance < other.cost + other.manhattan_distance

class AStarAgent(Agent):
    @override
    def __init__(self):
        self.memory: dict[CellPosition, AStarAgentMemoryCell] = {}
        self.plan_state: AStarAgentPlanningState = AStarAgentPlanningState.START
        self.expansion_state: AStarAgentExpansionState = AStarAgentExpansionState.NOT_EXPANDING
        self.current_path: list[CellPosition] = []
        self.open_queue: list[AStarAgentMemoryCell] = []
        self.visited: set[CellPosition] = set()
        self.expanding_cell: AStarAgentMemoryCell | None = None

    def _expand(self, last_result: TurnResult) -> list[Action]:
        # expand in up, right, down, left order
        if self.expansion_state == AStarAgentExpansionState.NOT_EXPANDING:
            self.expansion_state = AStarAgentExpansionState.UP
            return [Action.MOVE_UP, Action.WAIT, Action.WAIT, Action.WAIT, Action.WAIT]
        elif self.expansion_state == AStarAgentExpansionState.UP:
            self.expansion_state = AStarAgentExpansionState.RIGHT
            actions = [Action.MOVE_DOWN, Action.MOVE_RIGHT, Action.WAIT, Action.WAIT, Action.WAIT]
            if last_result.wall_hits == 1:
                actions[0] = Action.WAIT
            return actions
        elif self.expansion_state == AStarAgentExpansionState.RIGHT:
            self.expansion_state = AStarAgentExpansionState.DOWN
            actions = [Action.MOVE_LEFT, Action.MOVE_DOWN, Action.WAIT, Action.WAIT, Action.WAIT]
            if last_result.wall_hits == 1:
                actions[0] = Action.WAIT
            return actions
        elif self.expansion_state == AStarAgentExpansionState.DOWN:
            self.expansion_state = AStarAgentExpansionState.LEFT
            actions = [Action.MOVE_UP, Action.MOVE_LEFT, Action.WAIT, Action.WAIT, Action.WAIT]
            if last_result.wall_hits == 1:
                actions[0] = Action.WAIT
            return actions
        else:  # LEFT
            self.expansion_state = AStarAgentExpansionState.NOT_EXPANDING
            actions = [Action.MOVE_RIGHT, Action.WAIT, Action.WAIT, Action.WAIT, Action.WAIT]
            if last_result.wall_hits == 1:
                actions[0] = Action.WAIT
            return actions

    def _traverse(self, premade_actions: list[Action] | None = None) -> list[Action]:
        # move up to 5 steps toward next node, pad remainder with WAITs
        actions = premade_actions if premade_actions is not None else []
        while len(actions) < 5 and len(self.current_path) > 1:
            current_x, current_y = self.current_path.pop(0)
            next_x, next_y = self.current_path[0]
            if next_x > current_x:
                actions.append(Action.MOVE_RIGHT)
            elif next_x < current_x:
                actions.append(Action.MOVE_LEFT)
            elif next_y > current_y:
                actions.append(Action.MOVE_DOWN)
            else:
                actions.append(Action.MOVE_UP)

        if len(self.current_path) == 1:
            destination_pos = self.current_path[0]
            self.expanding_cell = self.memory[destination_pos]
            self.visited.add(destination_pos)
            self.plan_state = AStarAgentPlanningState.EXPANDING
            self.current_path = []
            if self.expanding_cell.pos == AStarAgentMemoryCell.goal_pos:
                print("solution found!!")
                self.expanding_cell.manhattan_distance = -100000 # offset the value of this cell so it is selected in the next traversal

        while len(actions) < 5:
            actions.append(Action.WAIT)

        return actions

    def _get_route_to_cell(self, start: AStarAgentMemoryCell, destination: AStarAgentMemoryCell) -> list[CellPosition]:
        journey_part_2: list[AStarAgentMemoryCell] = []
        curr: AStarAgentMemoryCell | None = destination
        while curr is not None:
            journey_part_2.append(curr)
            curr = curr.previous

        journey_part_2.reverse()

        journey_part_1: list[AStarAgentMemoryCell] = []
        curr = start
        while curr not in journey_part_2:
            journey_part_1.append(curr)
            curr = curr.previous

        path = journey_part_1 + journey_part_2[journey_part_2.index(curr):]
        return [cell.pos for cell in path]

    @override
    def plan_turn(self, last_result: TurnResult):
        # print(self.open_queue)
        if self.plan_state == AStarAgentPlanningState.EXPANDING:
            # On the very first expansion, initialize the starting cell from position feedback
            if self.expanding_cell is None:
                start_pos = CellPosition(*last_result.current_position)
                start_cell = self.memory[start_pos] if start_pos in self.memory else AStarAgentMemoryCell(CellState.EMPTY, start_pos, None)
                self.memory[start_pos] = start_cell
                self.visited.add(start_pos)
                self.expanding_cell = start_cell
                

            pos = CellPosition(*last_result.current_position)
            if self.expansion_state != AStarAgentExpansionState.NOT_EXPANDING \
                    and last_result.wall_hits == 0 and pos not in self.memory:
                new_cell = AStarAgentMemoryCell(CellState.EMPTY, pos, self.expanding_cell)
                self.memory[pos] = new_cell
                if self.expansion_state == AStarAgentExpansionState.UP:
                    new_cell.neighbors.down = self.expanding_cell
                    self.expanding_cell.neighbors.up = new_cell
                elif self.expansion_state == AStarAgentExpansionState.RIGHT:
                    new_cell.neighbors.left = self.expanding_cell
                    self.expanding_cell.neighbors.right = new_cell
                elif self.expansion_state == AStarAgentExpansionState.DOWN:
                    new_cell.neighbors.up = self.expanding_cell
                    self.expanding_cell.neighbors.down = new_cell
                elif self.expansion_state == AStarAgentExpansionState.LEFT:
                    new_cell.neighbors.right = self.expanding_cell
                    self.expanding_cell.neighbors.left = new_cell

                heappush(self.open_queue, new_cell)

            next_actions = self._expand(last_result)
            if self.expansion_state == AStarAgentExpansionState.NOT_EXPANDING:
                self.plan_state = AStarAgentPlanningState.TRAVERSING
                self.expanding_cell.fully_expanded = True
                cell_to_traverse_to = heappop(self.open_queue)
                while cell_to_traverse_to.fully_expanded and len(self.open_queue) > 0:
                    cell_to_traverse_to = heappop(self.open_queue)
                self.current_path = self._get_route_to_cell(self.expanding_cell, cell_to_traverse_to)
                return self._traverse(next_actions)

            return next_actions

        elif self.plan_state == AStarAgentPlanningState.START:
            self.plan_state = AStarAgentPlanningState.EXPANDING
            return [Action.WAIT for _ in range(5)]

        elif self.plan_state == AStarAgentPlanningState.TRAVERSING:
            return self._traverse()

    @override
    def reset_episode(self):
        if self.expanding_cell is not None:
            heappush(self.open_queue, self.expanding_cell)

        self.plan_state = AStarAgentPlanningState.TRAVERSING
        self.expansion_state = AStarAgentExpansionState.NOT_EXPANDING
        self.current_path = []
        self.visited = {pos for pos, cell in self.memory.items() if cell.fully_expanded}
        self.open_queue = [cell for cell in self.memory.values() if not cell.fully_expanded]
        self.expanding_cell = None

        heapify(self.open_queue)


        self.plan_state = AStarAgentPlanningState.START

class ManualMovementAgent(Agent):
    @override
    def __init__(self):
        self.memory: set[CellPosition] = {}
        self.next_action: Action

    @override
    def plan_turn(self, _):
        ans = self.next_action
        self.next_action = Action.WAIT
        return [ans]
    
    def set_action(self, action: Action):
        self.next_action = action
