import math
from typing import Tuple

from .pygame_mdp import X
from .pygame_mdp import Y
from .pygame_mdp import A

class Point:
    def __init__(self, x:float, y:float) -> None:
        self.x = x
        self.y = y

    def __call__(self) -> Tuple[int, int]:
        return (self.x, self.y)

class Rectangle:
    def __init__(self, topleft:Point, topright:Point, bottomright:Point, bottomleft:Point) -> None:
        self.topleft = topleft
        self.topright = topright
        self.bottomright = bottomright
        self.bottomleft = bottomleft

class Agent:
    width:int = 96 # width of ./assets/car.png
    height:int = 44 # height of ./assets/car.png
    view_width:int = 200
    view_height:int = 44 + 2*50
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

    def get_centre(self) -> Tuple[int, int]:
        return (self.get_x(), self.get_y())

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

    def is_point_in_view(self, view:Rectangle, point:Point) -> Tuple[bool, Point]:
        """
        Determines if the given point is inside the given view.

        Point is in view (i.e. point is inside polygon) if a ray from origin to point intesects the view edges an odd number of times.
            - Source: https://en.wikipedia.org/wiki/Point_in_polygon
        The ray intersects with the view edge based on the orientation of an ordered triplet of points in the plane.
            - Source: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

        Returns a tuple where element:
            0: true if the point is inside the view, false otherwise
            1: the relative position of the point inside the view
        """

        line_segments = [
            (view.topright, view.topleft),
            (view.topleft, view.bottomleft),
            (view.bottomleft, view.bottomright),
            (view.bottomright, view.topright)
        ]

        intersect_count = 0
        origin = Point(0, 0)
        for line_segment in line_segments:
            if self._do_intersect(line_segment[0], line_segment[1], origin, point):
                intersect_count += 1
        is_in_view = intersect_count > 0 and intersect_count % 2 != 0

        x = round(self._triangle_height(point, view.topright, view.topleft), self.precision)
        y = round(self._triangle_height(point, view.topright, view.bottomright), self.precision)
        relative_position = Point(x, y)

        return is_in_view, relative_position

    def _triangle_height(self, p:Point, q:Point, r:Point) -> float:
        """
        Determines the height of a triangle from the provided verticies using Heron's formula. The line segment 'qr' is the base.
        source: source: https://www.britannica.com/science/Herons-formula
        """
        a = math.sqrt((p.x - q.x) ** 2 + (p.y - q.y) ** 2)
        b = math.sqrt((p.x - r.x) ** 2 + (p.y - r.y) ** 2)
        c = math.sqrt((q.x - r.x) ** 2 + (q.y - r.y) ** 2)
        s = (a + b + c) / 2
        h = 2 * math.sqrt(s * (s - a) * (s - b) * (s - c)) / c
        return h

    def _orientation(self, p:Point, q:Point, r:Point) -> int:
        """
        Finds the orientation of an ordered triplet (p, q, r).

        Returns:
            0 --> p, q and r are collinear
            1 --> Clockwise
            2 --> Counterclockwise
        """
        # See https://www.geeksforgeeks.org/orientation-3-ordered-points/
        # for details of below formula.
        val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)

        return 0 if val == 0 else (1 if val > 0 else 2)

    def _on_segment(self, p:Point, q:Point, r:Point) -> bool:
        """
        Determines if point q lies on line segment 'pr' given three collinear points p, q, r
        """
        if q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y):
            return True

        return False

    def _do_intersect(self, p1:Point, q1:Point, p2:Point, q2:Point) -> bool:
        """
        Determines if line segment 'p1q1' and 'p2q2' intersect.
        """
        o1 = self._orientation(p1, q1, p2)
        o2 = self._orientation(p1, q1, q2)
        o3 = self._orientation(p2, q2, p1)
        o4 = self._orientation(p2, q2, q1)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special Cases
        # p1, q1 and p2 are collinear and p2 lies on segment p1q1
        if o1 == 0 and self._on_segment(p1, p2, q1):
            return True
    
        # p1, q1 and q2 are collinear and q2 lies on segment p1q1
        if o2 == 0 and self._on_segment(p1, q2, q1):
            return True
    
        # p2, q2 and p1 are collinear and p1 lies on segment p2q2
        if o3 == 0 and self._on_segment(p2, p1, q2):
            return True
    
        # p2, q2 and q1 are collinear and q1 lies on segment p2q2
        if o4 == 0 and self._on_segment(p2, q1, q2):
            return True
    
        return False
    
    def _get_rotation_x(self, radius:float, angle:float) -> float:
        return round(self.get_x() + radius * math.cos(angle * math.pi / 180), self.precision)

    def _get_rotation_y(self, radius:float, angle:float) -> float:
        return round(self.get_y() - radius * math.sin(angle * math.pi / 180), self.precision)
