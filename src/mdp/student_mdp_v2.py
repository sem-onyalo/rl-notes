import random
from typing import Tuple

class StudentMDPV2:
    """
    This class represents the student MDP example.
    """

    def __init__(self):
        self.n_states = 5 # TikTok, Class1, Class2, Class3, End
        self.n_actions = 5 # TikTok, Quit, Study, Sleep, Pub
        self.terminal_state = 4

        self.state_actions = {
            # schema: { state: { action: (reward, [next_state]) } }
            0: {
                0: (-1, [0]),
                1: ( 0, [1])
            },
            1: {
                0: (-1, [0]),
                2: (-2, [2])
            },
            2: {
                2: (-2, [3]),
                3: ( 0, [4])
            },
            3: {
                2: (10, [4]),
                4: ( 1, [1, 2, 3])
            },
            4: {
                0: (0, [4]),
                1: (0, [4]),
                2: (0, [4]),
                3: (0, [4]),
                4: (0, [4]),
            } # terminal state, i.e. all actions give zero rewards result to being in the same state
        }

        self.action_probabilities = {
            # schema: {state: {next_state: probability}}
            3: {
                1: 0.2,
                2: 0.4,
                3: 0.4
            }
        }

    def start(self) -> float:
        self.current_state = 0
        return self.current_state

    def step(self, action:int) -> Tuple[float, int, bool]:
        reward, next_state = self.get_reward_and_next_state(action)
        is_terminal = next_state == self.terminal_state
        self.current_state = next_state
        return reward, next_state, is_terminal

    def get_reward_and_next_state(self, action:int) -> Tuple[float, int]:
        if not action in self.state_actions[self.current_state]:
            reward = 0
            next_state = self.current_state
        else:
            reward, next_states = self.state_actions[self.current_state][action]

            if len(next_states) > 1:
                probability_total = 0
                probability_value = random.random()
                for possible_next_state in self.action_probabilities[self.current_state]:
                    probability_total += self.action_probabilities[self.current_state][possible_next_state]
                    if probability_value < probability_total:
                        next_state = possible_next_state
                        break
            else:
                next_state = next_states[0]

        return reward, next_state
