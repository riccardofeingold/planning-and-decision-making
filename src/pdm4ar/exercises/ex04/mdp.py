from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from pdm4ar.exercises.ex04.structures import Action, Policy, State, ValueFunc, Cell


class GridMdp:
    def __init__(self, grid: NDArray[np.int64], gamma: float = 0.9):
        assert len(grid.shape) == 2, "Map is invalid"
        self.grid = grid
        """The map"""
        self.gamma: float = gamma
        """Discount factor"""

        # additional useful variables
        x, y = np.where(self.grid == Cell.START)
        self.start = (int(x), int(y))

        x, y = np.where(self.grid == Cell.GOAL)
        self.goal = (int(x), int(y))


    # check if state is inside map
    def is_state_valid(self, state: State, action: Action) -> bool:
        height, width = self.grid.shape
        x_next_state, y_next_state = self.get_next_state_given_action(action=action, current_state=state)

        if x_next_state < 0 or x_next_state >= height or y_next_state < 0 or y_next_state >= width:
            return False
        else:
            return True
    
    # get next state by following the action
    def get_next_state_given_action(self, action: Action, current_state: State) -> State:
        if action == Action.ABANDON:
            return self.start
        elif action == Action.STAY:
            return current_state
        elif action == Action.NORTH:
            return (current_state[0] - 1, current_state[1])
        elif action == Action.EAST:
            return (current_state[0], current_state[1] + 1)
        elif action == Action.SOUTH:
            return (current_state[0] + 1, current_state[1])
        elif action == Action.WEST:
            return (current_state[0], current_state[1] - 1)

    # get all the relevant next states
    def get_next_states(self, current_state: State) -> set(State):
        next_states = set()

        ## if current state is goal, the only possible next state
        ## is GOAL
        if self.grid[current_state] == Cell.GOAL:
            next_states.add(current_state)
            return next_states

        # add start state
        next_states.add(self.start)

        # add current_state only if swamp
        if self.grid[current_state] == Cell.SWAMP:
            next_states.add(current_state)

        # add next states based on action
        ## EAST
        next_states.add((current_state[0], current_state[1] + 1))
        ## WEST
        next_states.add((current_state[0], current_state[1] - 1))
        ## SOUTH
        next_states.add((current_state[0] + 1, current_state[1]))
        ## NORTH
        next_states.add((current_state[0] - 1, current_state[1]))

        return next_states
    
    # get only the allowed actions that keep the robot inside the map
    def get_allowed_actions(self, current_state: State) -> set(Action):
        actions = set()

        if self.grid[current_state] == Cell.GOAL:
            actions.add(Action.STAY)
            return actions

        ## look up table
        action_list = [
            Action.NORTH,
            Action.EAST,
            Action.SOUTH,
            Action.WEST,
            Action.ABANDON
        ]
        for a in action_list:
            ## only return actions that keep the robot inside the map
            if self.is_state_valid(state=current_state, action=a):
                actions.add(a)
        
        return actions
    
    def convert_outsiders_to_insiders(self, state: State) -> State:
        if self.is_state_valid(state, Action.STAY) == False:
            return self.start
        else:
            return state
    
    def get_transition_prob(self, state: State, action: Action, next_state: State) -> float:
        """Returns P(next_state | state, action)"""
        # todo
        current_cell_type = self.grid[state]
        # this will also give me states that could be outside of the map
        next_state_following_a = self.get_next_state_given_action(action=action, current_state=state)
        
        ## check if next_state is in map
        converted_state = self.convert_outsiders_to_insiders(next_state)
        if converted_state == next_state:
            is_next_state_in_map = True
        else:
            is_next_state_in_map = False
        
        ## get the next cell type
        next_cell_type = self.grid[converted_state]

        # ABANDON
        if action == Action.ABANDON:
            # next state has to be inside the map to be the "true" start state
            # if not then it's a start that would be caused by going out of the map
            # this case cannot be consider by ABANDON since this action cannot move the robot
            # out of the map
            if next_cell_type == Cell.START and is_next_state_in_map:
                if current_cell_type != Cell.GOAL:
                    return 1.0
            return 0
        
        # STAY
        if action == Action.STAY:
            if current_cell_type == Cell.GOAL and next_cell_type == Cell.GOAL:
                return 1.0
            return 0
       
        # NORTH, SOUTH, EAST, WEST
        if action == Action.NORTH or action == Action.SOUTH or action == Action.EAST or action == Action.WEST:
            if current_cell_type == Cell.GRASS or current_cell_type == Cell.START:
                if next_state_following_a == next_state:
                    return 0.75
                
                if next_cell_type == Cell.START:
                    if is_next_state_in_map:
                        first_value, second_value = tuple(map(lambda i, j: abs(i - j), next_state, state))
                        distance_squared = int(first_value*first_value + second_value*second_value)

                        if distance_squared != 1:
                            return 0
                return 0.25/3
            elif current_cell_type == Cell.SWAMP:
                if next_state_following_a == next_state:
                    return 0.5
                elif next_state == state:
                    return 0.2
                elif next_cell_type == Cell.START and is_next_state_in_map:
                    return 0.05
                else:
                    return 0.25/3

        return 0


    def stage_reward(self, state: State, action: Action, next_state: State) -> float:
        # todo
        ## get the current cell type 
        current_cell_type = self.grid[state]

        ## check if next_state is inside the map
        converted_state = self.convert_outsiders_to_insiders(next_state)
        if converted_state == next_state:
            is_next_state_in_map = True
        else:
            is_next_state_in_map = False
        
        ## get the next cell type
        next_cell_type = self.grid[converted_state]

        ## cell type == GOAL
        if current_cell_type == Cell.GOAL:
            return 50
        
        ## if action == ABANDON then cost = -10
        ## ABANDON can only be call in non-GOAL states
        if action == Action.ABANDON:
            return -10

        ## cell type == GRASS
        if current_cell_type == Cell.GRASS or current_cell_type == Cell.START:
            # here we could get either -1 for a normal transition
            # or - 11 for a transition to the start when robot is at edge or coner
            # or it is -10 if action == ABANDON
            if next_cell_type == Cell.START and is_next_state_in_map == False:
                return -11
            else:
                return -1

        ## cell type == SWAMP
        if current_cell_type == Cell.SWAMP:
            # here we could get either -2 for a normal transition
            # or -12 for a transition to the start when robot is at edge or corner or by chance 0.05
            # or -10 through ABANDON (decided) 
            if next_cell_type == Cell.START:
                return -12
            else:
                return -2
                
        return 0


class GridMdpSolver(ABC):
    @staticmethod
    @abstractmethod
    def solve(grid_mdp: GridMdp) -> Tuple[ValueFunc, Policy]:
        pass
