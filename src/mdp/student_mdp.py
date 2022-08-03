import logging
import random

class StudentMDP:
    """
    This class represents the student MDP example.
    """

    def __init__(self):
        self.states = list(range(5)) # TikTok, Class1, Class2, Class3, End
        self.actions = list(range(5)) # TikTok, Quit, Study, Sleep, Pub
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

    def __eq__(self, other):
        if other != None:
            return (
                self.states == other.states
                and self.actions == other.actions
                and self.terminal_state == other.terminal_state
                and self.state_actions == other.state_actions
                and self.action_probabilities == other.action_probabilities
            )

        return False

    def __getitem__(self, i):
        return self.state_actions[i]

    def is_terminal_state(self, state):
        return state == self.terminal_state

    def get_initial_state(self):
        return 0

    def get_state_actions(self):
        return self.state_actions

    def get_reward_and_next_state(self, state, action):
        if self.is_terminal_state(state):
            is_terminal_state = True
            next_state = state
            reward = 0
        else:
            is_terminal_state = False
            reward, next_states = self.state_actions[state][action]
            if len(next_states) > 1:
                probability_total = 0
                probability_value = random.random()
                for possible_next_state in self.action_probabilities[state]:
                    probability_total += self.action_probabilities[state][possible_next_state]
                    if probability_value < probability_total:
                        next_state = possible_next_state
                        break
            else:
                next_state = next_states[0]

        return reward, next_state, is_terminal_state

    def get_reward_and_next_states(self, state, action):
        """
        Gets the result of the specified action when in the specified state.
        :returns: reward, next_states, is_terminal_state OR None if state-action pair DNE.
        """
        if state in self.state_actions and action in self.state_actions[state]:
            reward, next_states = self.state_actions[state][action]
            return reward, next_states, self.is_terminal_state(state)

        return None

    def get_action_probability(self, state, next_state):
        """
        This _might_ get pull out of the MDP and turn into a policy function.
        """
        if state in self.action_probabilities and next_state in self.action_probabilities[state]:
            return self.action_probabilities[state][next_state]
        else:
            return 1 # if probability isn't explicitly set then always take that action (i.e. act greedily)

