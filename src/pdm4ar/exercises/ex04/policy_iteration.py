from typing import Tuple

import numpy as np
from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver, Action
from pdm4ar.exercises.ex04.structures import ValueFunc, Policy
from pdm4ar.exercises_def.ex04.utils import time_function


class PolicyIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> Tuple[ValueFunc, Policy]:
        value_func = np.zeros_like(grid_mdp.grid).astype(float)
        policy = np.zeros_like(grid_mdp.grid).astype(int)

        # todo implement here
        ## adjusting initial condition
        value_func[grid_mdp.goal] = 500
        policy[grid_mdp.goal] = Action.STAY
        
        ## Implementing the modified Policy Iteration Algorithm
        while True:
            ## Policy evaluation
            height, width = grid_mdp.grid.shape
            epsilon = 1e-2
            max_error = float('inf')

            while max_error > epsilon:
                # iterate through each cell and update the value
                error_list = []
                for row in range(height):
                    for col in range(width):
                        current_state = (row, col)
                        current_v = value_func[current_state]
                        # returns only the relevant next_states
                        # this would be the four neighbours, START, and the current state itself
                        next_states = grid_mdp.get_next_states(current_state)
                        action = policy[current_state]

                        value = 0
                        for next_state in next_states:
                            T = grid_mdp.get_transition_prob(current_state, action, next_state)

                            if T != 0:
                                value += T * (grid_mdp.stage_reward(current_state, action, next_state) + grid_mdp.gamma * value_func[grid_mdp.convert_outsiders_to_insiders(next_state)])
                    
                        value_func[current_state] = value

                        error_list.append(np.abs(value_func[current_state] - current_v))

                max_error = np.max(error_list)

            # Policy Improvement
            policy_is_stable = True
            for row in range(height):
                for col in range(width):
                    current_state = (row, col)
                    action_old_policy = policy[current_state]
                    next_states = grid_mdp.get_next_states(current_state)
                    allowed_actions = grid_mdp.get_allowed_actions(current_state)

                    best_value = float('-inf')
                    best_action = None
                    
                    for a in allowed_actions:
                        value = 0
                        for next_state in next_states:
                            T = grid_mdp.get_transition_prob(current_state, a, next_state)

                            if T != 0:
                                value += T * (grid_mdp.stage_reward(current_state, a, next_state) + grid_mdp.gamma * value_func[grid_mdp.convert_outsiders_to_insiders(next_state)])
                        if value > best_value:
                            best_value = value
                            best_action = a

                    policy[current_state] = best_action

                    if action_old_policy != best_action:
                        policy_is_stable = False

            if policy_is_stable:
                return value_func, policy
