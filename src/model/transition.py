class Transition:
    def __init__(self, state, action, next_state, reward) -> None:
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, self.__class__)
            and __o.state == self.state
            and __o.action == self.action
            and __o.next_state == self.next_state
            and __o.reward == self.reward
        )

    def to_tuple(self):
        return (self.state, self.action, self.next_state, self.reward)
