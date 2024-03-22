from typing import Tuple

import numpy as np
from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver, Action, Cell
from pdm4ar.exercises.ex04.structures import Policy, ValueFunc
from pdm4ar.exercises_def.ex04.utils import time_function


class ValueIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> Tuple[ValueFunc, Policy]:
        value_func = np.zeros_like(grid_mdp.grid).astype(float)
        policy = np.zeros_like(grid_mdp.grid).astype(int)

        # todo implement here
        ## adjusting initialization of value_func and policy
        value_func[grid_mdp.goal] = 500
        policy[grid_mdp.goal] = Action.STAY

        height, width = grid_mdp.grid.shape
        epsilon = 1e-2
        max_error = float('inf')

        # optimise value func
        while max_error > epsilon:
            # iterate through each cell and update the value
            # Uncomment below for standard VI
            ## value_func_copy = value_func.copy()
            error_list = []
            for row in range(height):
                for col in range(width):
                    current_state = (row, col)
                    # Uncomment for standard VI
                    ## current_v = value_func_copy[current_state]
                    current_v = value_func[current_state]
                    best_value = float('-inf')
                    
                    # returns only the relevant next_states
                    # this would be the four neighbours, START, and the current state itself
                    next_states = grid_mdp.get_next_states(current_state)
                    allowed_actions = grid_mdp.get_allowed_actions(current_state)
                    for a in allowed_actions:
                        value = 0

                        for next_state in next_states:
                            T = grid_mdp.get_transition_prob(current_state, a, next_state)

                            if T != 0:
                                value += T * (grid_mdp.stage_reward(current_state, a, next_state) + grid_mdp.gamma * value_func[grid_mdp.convert_outsiders_to_insiders(next_state)])

                        if value > best_value:
                            best_value = value
                            best_action = a
                    
                    value_func[current_state] = best_value
                    policy[current_state] = best_action
                    error_list.append(np.abs(value_func[current_state] - current_v))

            max_error = np.max(error_list)
            
        return value_func, policy