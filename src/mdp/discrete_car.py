import math
from typing import List
from typing import Tuple

from .pygame_mdp import X
from .pygame_mdp import Y
from .pygame_mdp import A
from model import Point
from model import Rectangle
from model import SensorArray
from model import Sensor

class Agent:
    width:int = 96 # width of ./assets/car.png
    height:int = 44 # height of ./assets/car.png
    view_width:int = 100
    view_height:int = 100
    collision_sensor_size:int = 10
    precision:int = 8
    def __init__(self, position) -> None:
        self.position = position
        self.position_history = []
        self.position_history.append(position)
        self.angle_step = 5
        self.forward_step = 5

    def get_position(self) -> Tuple[int, int, int]:
        return self.position

    def update_position(self, position:Tuple[int, int, int]) -> None:
        self.position = position
        self.position_history.append(self.position)

    def reset_position(self, position:Tuple[int, int, int]) -> None:
        self.position_history.clear()
        self.update_position(position)

    def get_x(self) -> int:
        return self.position[X]

    def get_y(self) -> int:
        return self.position[Y]

    def get_a(self) -> int:
        return self.position[A]

    def get_viewpoint(self) -> Point:
        radius = self.width//2
        x = self._get_rotation_x(radius, self.get_a())
        y = self._get_rotation_y(radius, self.get_a())
        return Point(x, y)

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

    def get_sensor(self) -> SensorArray:
        front_view = self._get_view()
        left_sensor = self._get_left_sensor()
        right_view = self._get_right_view()
        return SensorArray(front_view, left_sensor, right_view)

    def _get_view(self) -> Rectangle:
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

    def _get_left_view(self) -> Rectangle:
        radius_topleft = math.sqrt((self.width//2)**2 + (self.height//2 + self.collision_sensor_size)**2)
        angle_topleft = math.atan2(self.height//2 + self.collision_sensor_size, self.width//2)
        topleft_x = self._get_rotation_x(radius_topleft, self.get_a() + (angle_topleft * 180 / math.pi))
        topleft_y = self._get_rotation_y(radius_topleft, self.get_a() + (angle_topleft * 180 / math.pi))

        radius_topright = math.sqrt((self.width//2 + self.collision_sensor_size)**2 + (self.height//2 + self.collision_sensor_size)**2)
        angle_topright = math.atan2(self.height//2 + self.collision_sensor_size, self.width//2 + self.collision_sensor_size)
        topright_x = self._get_rotation_x(radius_topright, self.get_a() + (angle_topright * 180 / math.pi))
        topright_y = self._get_rotation_y(radius_topright, self.get_a() + (angle_topright * 180 / math.pi))

        radius_bottomright = math.sqrt((self.width//2 + self.collision_sensor_size)**2 + (self.height//2)**2)
        angle_bottomright = math.atan2(self.height//2, self.width//2 + self.collision_sensor_size)
        bottomright_x = self._get_rotation_x(radius_bottomright, self.get_a() + (angle_bottomright * 180 / math.pi))
        bottomright_y = self._get_rotation_y(radius_bottomright, self.get_a() + (angle_bottomright * 180 / math.pi))

        radius_bottomleft = round(math.sqrt((self.width//2)**2 + (self.height//2)**2))
        angle_bottomleft = math.atan2(self.height//2, self.width//2)
        bottomleft_x = self._get_rotation_x(radius_bottomleft, self.get_a() + (angle_bottomleft * 180 / math.pi))
        bottomleft_y = self._get_rotation_y(radius_bottomleft, self.get_a() + (angle_bottomleft * 180 / math.pi))

        return Rectangle(Point(topleft_x, topleft_y), Point(topright_x, topright_y), Point(bottomright_x, bottomright_y), Point(bottomleft_x, bottomleft_y))

    def _get_left_sensor(self) -> Sensor:
        points = []
        added_points = {}
        anchor = Point(self.get_x() + self.width//2, self.get_y() + self.height//2)
        x_start = round(anchor.x)
        x_end = round(anchor.x + self.collision_sensor_size + 1)
        for x in range(x_start, x_end):
            y_start = round(anchor.y - self.collision_sensor_size)
            y_end = round(anchor.y + 1)
            for y in range(y_start, y_end):
                x_length = self.width//2 + x - anchor.x
                y_length = self.height//2 + anchor.y - y
                radius = math.sqrt((x_length)**2 + (y_length)**2)
                angle = math.atan2(y_length, x_length)
                x_rotated = self._get_rotation_x(radius, self.get_a() + (angle * 180 / math.pi))
                y_rotated = self._get_rotation_y(radius, self.get_a() + (angle * 180 / math.pi))
                point = (round(x_rotated), round(y_rotated))
                if not point in added_points:
                    added_points[point] = Point(point[X], point[Y])
                    points.append(added_points[point])

        anchor_radius = math.sqrt(anchor.x**2 + anchor.y**2)
        anchor_angle = math.atan2(anchor.y, anchor.x)
        x_anchor_rotated = self._get_rotation_x(anchor_radius, self.get_a() + (anchor_angle * 180 / math.pi))
        y_anchor_rotated = self._get_rotation_y(anchor_radius, self.get_a() + (anchor_angle * 180 / math.pi))
        anchor_rotated = Point(round(x_anchor_rotated), round(y_anchor_rotated))
        print(anchor_rotated())
        quit()
        return Sensor(anchor_rotated, points)

    def _get_right_view(self) -> Rectangle:
        radius_topleft = math.sqrt((self.width//2)**2 + (self.height//2 + self.collision_sensor_size)**2)
        angle_topleft = -math.atan2(self.height//2 + self.collision_sensor_size, self.width//2)
        topleft_x = self._get_rotation_x(radius_topleft, self.get_a() + (angle_topleft * 180 / math.pi))
        topleft_y = self._get_rotation_y(radius_topleft, self.get_a() + (angle_topleft * 180 / math.pi))

        radius_topright = math.sqrt((self.width//2 + self.collision_sensor_size)**2 + (self.height//2 + self.collision_sensor_size)**2)
        angle_topright = -math.atan2(self.height//2 + self.collision_sensor_size, self.width//2 + self.collision_sensor_size)
        topright_x = self._get_rotation_x(radius_topright, self.get_a() + (angle_topright * 180 / math.pi))
        topright_y = self._get_rotation_y(radius_topright, self.get_a() + (angle_topright * 180 / math.pi))

        radius_bottomright = math.sqrt((self.width//2 + self.collision_sensor_size)**2 + (self.height//2)**2)
        angle_bottomright = -math.atan2(self.height//2, self.width//2 + self.collision_sensor_size)
        bottomright_x = self._get_rotation_x(radius_bottomright, self.get_a() + (angle_bottomright * 180 / math.pi))
        bottomright_y = self._get_rotation_y(radius_bottomright, self.get_a() + (angle_bottomright * 180 / math.pi))

        radius_bottomleft = round(math.sqrt((self.width//2)**2 + (self.height//2)**2))
        angle_bottomleft = -math.atan2(self.height//2, self.width//2)
        bottomleft_x = self._get_rotation_x(radius_bottomleft, self.get_a() + (angle_bottomleft * 180 / math.pi))
        bottomleft_y = self._get_rotation_y(radius_bottomleft, self.get_a() + (angle_bottomleft * 180 / math.pi))

        return Rectangle(Point(topleft_x, topleft_y), Point(topright_x, topright_y), Point(bottomright_x, bottomright_y), Point(bottomleft_x, bottomleft_y))

    def _get_rotation_x(self, radius:float, angle:float) -> float:
        return round(self.get_x() + radius * math.cos(angle * math.pi / 180), self.precision)

    def _get_rotation_y(self, radius:float, angle:float) -> float:
        return round(self.get_y() - radius * math.sin(angle * math.pi / 180), self.precision)
