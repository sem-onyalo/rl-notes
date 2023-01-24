from typing import Tuple

class Actor:
    def __init__(self, size, colour, position) -> None:
        self.size = size
        self.colour = colour
        self.position = position
        self.position_history = []
        self.position_history.append(position)

    def get_position(self) -> Tuple[int, int]:
        return self.position

    def update_position(self, position:Tuple[int, int]) -> None:
        self.position = position
        self.position_history.append(self.position)

    def get_x(self) -> int:
        return self.position[0]

    def get_y(self) -> int:
        return self.position[1]

    def get_position_idx(self) -> Tuple[int, int]:
        return (self.get_x() - 1, self.get_y() - 1)