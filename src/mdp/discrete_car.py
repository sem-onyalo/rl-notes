import math
from typing import Tuple

from .pygame_mdp import X
from .pygame_mdp import Y
from .pygame_mdp import A
from model import Point
from model import Rectangle

class Agent:
    width:int = 96 # width of ./assets/car.png
    height:int = 44 # height of ./assets/car.png
    view_width:int = 100
    view_height:int = 100
    precision:int = 2
    def __init__(self, position) -> None:
        self.position = position
        self.position_history = []
        self.position_history.append(position)
        self.angle_step = 5
        self.forward_step = 1

    def get_position(self) -> Tuple[int, int, int]:
        return self.position

    def update_position(self, position:Tuple[int, int, int]) -> None:
        self.position = position
        self.position_history.append(self.position)

    def get_x(self) -> int:
        return self.position[X]

    def get_y(self) -> int:
        return self.position[Y]

    def get_a(self) -> int:
        return self.position[A]

    def get_centre(self) -> Point:
        return Point(self.get_x(), self.get_y())

    def turn_left(self) -> None:
        new_angle = 0 if self.get_a() + self.angle_step >= 360 else self.get_a() + self.angle_step
        self.update_position((self.get_x(), self.get_y(), new_angle))

    def turn_right(self) -> None:
        new_angle = (360 - self.angle_step) if self.get_a() - self.angle_step < 0 else self.get_a() - self.angle_step
        self.update_position((self.get_x(), self.get_y(), new_angle))

    def forward(self) -> None:
        new_x = self._get_rotation_x(self.forward_step, self.get_a())
        new_y = self._get_rotation_y(self.forward_step, self.get_a())
        self.update_position((new_x, new_y, self.get_a()))

    def get_polygon(self) -> Rectangle:
        radius = round(math.sqrt((self.width//2)**2 + (self.height//2)**2))

        angle_topright = math.atan2(self.height//2, self.width//2)
        angle_topleft = -angle_topright + math.pi
        angle_bottomleft = angle_topright + math.pi
        angle_bottomright = -angle_topright

        topright_x = self._get_rotation_x(radius, self.get_a() + (angle_topright * 180 / math.pi))
        topright_y = self._get_rotation_y(radius, self.get_a() + (angle_topright * 180 / math.pi))

        topleft_x = self._get_rotation_x(radius, self.get_a() + (angle_topleft * 180 / math.pi))
        topleft_y = self._get_rotation_y(radius, self.get_a() + (angle_topleft * 180 / math.pi))

        bottomleft_x = self._get_rotation_x(radius, self.get_a() + (angle_bottomleft * 180 / math.pi))
        bottomleft_y = self._get_rotation_y(radius, self.get_a() + (angle_bottomleft * 180 / math.pi))

        bottomright_x = self._get_rotation_x(radius, self.get_a() + (angle_bottomright * 180 / math.pi))
        bottomright_y = self._get_rotation_y(radius, self.get_a() + (angle_bottomright * 180 / math.pi))

        return Rectangle(Point(topleft_x, topleft_y), Point(topright_x, topright_y), Point(bottomright_x, bottomright_y), Point(bottomleft_x, bottomleft_y))

    def get_view(self) -> Rectangle:
        radius_short = math.sqrt((self.width//2)**2 + (self.view_height//2)**2)
        radius_long = math.sqrt((self.width//2 + self.view_width)**2 + (self.view_height//2)**2)

        angle_topright = math.atan2(self.view_height//2, self.width//2 + self.view_width)
        angle_topleft = math.atan2(self.view_height//2, self.width//2)
        angle_bottomleft = -angle_topleft
        angle_bottomright = -angle_topright

        topright_x = self._get_rotation_x(radius_long, self.get_a() + (angle_topright * 180 / math.pi))
        topright_y = self._get_rotation_y(radius_long, self.get_a() + (angle_topright * 180 / math.pi))

        topleft_x = self._get_rotation_x(radius_short, self.get_a() + (angle_topleft * 180 / math.pi))
        topleft_y = self._get_rotation_y(radius_short, self.get_a() + (angle_topleft * 180 / math.pi))

        bottomleft_x = self._get_rotation_x(radius_short, self.get_a() + (angle_bottomleft * 180 / math.pi))
        bottomleft_y = self._get_rotation_y(radius_short, self.get_a() + (angle_bottomleft * 180 / math.pi))

        bottomright_x = self._get_rotation_x(radius_long, self.get_a() + (angle_bottomright * 180 / math.pi))
        bottomright_y = self._get_rotation_y(radius_long, self.get_a() + (angle_bottomright * 180 / math.pi))

        return Rectangle(Point(topleft_x, topleft_y), Point(topright_x, topright_y), Point(bottomright_x, bottomright_y), Point(bottomleft_x, bottomleft_y))

    def _get_rotation_x(self, radius:float, angle:float) -> float:
        return round(self.get_x() + radius * math.cos(angle * math.pi / 180), self.precision)

    def _get_rotation_y(self, radius:float, angle:float) -> float:
        return round(self.get_y() - radius * math.sin(angle * math.pi / 180), self.precision)
