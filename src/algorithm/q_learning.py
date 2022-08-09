import logging

class QLearning:
    """
    This class represents the Q-Learning off-policy control algorithm.
    """

    def __init__(self, mdp, function, policy, discount_rate, change_rate, max_episodes=100):
        self.mdp = mdp
        self.function = function
        self.policy = policy
        self.discount_rate = discount_rate
        self.change_rate = change_rate
        self.max_episodes = max_episodes

    def __eq__(self, other):
        if other != None:
            return (
                self.mdp == other.mdp
                and self.function == other.function
                and self.policy == other.policy
                and self.discount_rate == other.discount_rate
                and self.change_rate == other.change_rate
                and self.max_episodes == other.max_episodes)

        return False

    def run(self):
        for i in range(1, self.max_episodes + 1):
            logging.info(f"Episode {i}")

            path = list()
            is_terminal = False
            state = self.mdp.get_initial_state()
            while not is_terminal:
                action = self.policy.get_action(state)
                reward, next_state, is_terminal = self.mdp.get_reward_and_next_state(state, action)

                max_action_value = None
                for next_action in self.mdp[next_state]:
                    value = self.function(next_state, next_action)
                    if max_action_value == None or value > max_action_value:
                        max_action_value = value

                # Q(S,A) = Q(S,A) + a * (R + y * max_a[Q(S',a)] - Q(S,A))
                current_value = self.function(state, action)
                new_value = current_value + self.change_rate * (reward + (self.discount_rate * max_action_value) - current_value)
                self.function.update_value(state, action, new_value)

                # update policy
                optimal_action = self.function.get_optimal_action(state)
                if action != optimal_action:
                    self.policy.update_policy(state, optimal_action)

                path.append(state)
                state = next_state

            logging.debug(f"path: {path}")
            logging.info(f"function: {self.function}")

        logging.info(f"policy: {self.policy}")
        return self.function, self.policy

