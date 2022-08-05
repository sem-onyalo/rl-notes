import logging

class MonteCarlo:
    """
    This class represents the first-visit Monte Carlo exploring starts control (policy optimization) algorithm.
    """

    def __init__(self, mdp, function, policy, discount_rate, max_episodes=100, do_glie=True):
        self.mdp = mdp
        self.function = function
        self.policy = policy
        self.discount_rate = discount_rate
        self.max_episodes = max_episodes
        self.do_glie = do_glie

    def __eq__(self, other):
        if other != None:
            return (
                self.mdp == other.mdp
                and self.function == other.function
                and self.policy == other.policy
                and self.discount_rate == other.discount_rate
                and self.max_episodes == other.max_episodes
                and self.do_glie == other.do_glie)

        return False

    def run(self):
        total_visits = {
            state: {
                action: 0
                for action in self.mdp[state]
            } for state in self.mdp.get_state_actions()
        }

        for i in range(1, self.max_episodes + 1):
            logging.info(f"Episode {i}")

            episode_visit = {
                state: {
                    action: False
                    for action in self.mdp[state]
                } for state in self.mdp.get_state_actions()
            }

            returns = {
                state: {
                    action: 0.
                    for action in self.mdp[state]
                } for state in self.mdp.get_state_actions()
            }

            path = list()
            is_terminal = False
            state = self.mdp.get_initial_state()
            while not is_terminal:
                action = self.policy.get_action(state)
                reward, next_state, is_terminal = self.mdp.get_reward_and_next_state(state, action)
                total_visits[state][action] += 1

                if not episode_visit[state][action]:
                    episode_visit[state][action] = True

                # update the total rewards for all the state-action pairs that have been taken in this episode
                # TODO: pretty sure I need to move this out of the episodes. I don't think the monte-carlo algorithm
                #       updates the action-value function within the episode (i.e. like TD). Move right before policy 
                #       update
                for state_visited in returns:
                    if returns[state_visited] != None:
                        for action_taken in returns[state_visited]:
                            if episode_visit[state_visited][action_taken]:
                                # calculate the sum of all discounted rewards (i.e. G = y * G + R[t+1])
                                total_reward = self.discount_rate * returns[state_visited][action_taken] + reward

                                # calculate the incremental mean of the action-value function
                                # (i.e. Q(S,A) = Q(S,A) + 1/N(S,A) * (Gt - Q(S,A)))
                                current_value = self.function(state_visited, action_taken)
                                visit_count = total_visits[state_visited][action_taken]
                                new_value = current_value + (1/visit_count * (total_reward - current_value))
                                returns[state_visited][action_taken] = round(new_value, 1)
                                self.function.update_value(state_visited, action_taken, new_value)

                path.append(state)
                state = next_state

            # update policy
            if self.do_glie:
                self.policy.update_epsilon(1/i)

            for state in self.mdp.get_state_actions():
                if not self.mdp.is_terminal_state(state):
                    optimal_action = self.function.get_optimal_action(state)
                    self.policy.update_policy(state, optimal_action)

            logging.debug(f"path: {path}")
            logging.debug(f"total visits: {total_visits}")
            logging.debug(f"returns: {returns}")
            logging.info(f"function: {self.function}")

        logging.info(f"policy: {self.policy}")
        return self.function, self.policy

