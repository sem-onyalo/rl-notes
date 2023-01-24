from typing import List

from .geometry import distance
from .rectangle import Point
from .rectangle import Rectangle

class Sensor:
    def __init__(self, anchor_point:Point, points:List[Point]) -> None:
        self.anchor_point = anchor_point
        self.points = points
    def get_distance(self, points:List[Point]) -> float:
        assert isinstance(points, list)
        assert all([isinstance(p, Point) for p in points])
        assert len(points) > 0

        distance_min = None
        for point in points:
            distance_cur = distance(self.anchor_point, point)
            distance_min = distance_cur if distance_min == None or distance_cur < distance_min else distance_min
        return distance_min

class SensorArray:
    front:Rectangle
    left:Sensor
    right:Rectangle

    def __init__(self, front:Rectangle, left:Sensor, right:Rectangle) -> None:
        self.front = front
        self.left = left
        self.right = right
