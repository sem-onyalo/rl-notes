import math
from typing import Tuple

from .point import Point
from .rectangle import Rectangle

PRECISION = 2

def distance(p:Point, q:Point) -> float:
    return math.sqrt((q.x - p.x)**2 + (q.y - p.y)**2)

def is_point_in_polygon_with_position(polygon:Rectangle, point:Point) -> Tuple[bool, Point]:
    is_in_polygon = is_point_in_polygon(polygon, point)

    x = round(triangle_height(point, polygon.topright, polygon.topleft), PRECISION)
    y = round(triangle_height(point, polygon.topright, polygon.bottomright), PRECISION)
    relative_position = Point(x, y)

    return is_in_polygon, relative_position

def is_point_in_polygon(polygon:Rectangle, point:Point) -> bool:
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
        (polygon.topright, polygon.topleft),
        (polygon.topleft, polygon.bottomleft),
        (polygon.bottomleft, polygon.bottomright),
        (polygon.bottomright, polygon.topright)
    ]

    intersect_count = 0
    ray_start = Point(0, point.y)
    for line_segment in line_segments:
        if do_intersect(line_segment[0], line_segment[1], ray_start, point):
            intersect_count += 1
    return intersect_count > 0 and intersect_count % 2 != 0

def triangle_height(p:Point, q:Point, r:Point) -> float:
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

def orientation(p:Point, q:Point, r:Point) -> int:
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

def on_segment(p:Point, q:Point, r:Point) -> bool:
    """
    Determines if point q lies on line segment 'pr' given three collinear points p, q, r
    """
    if q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y):
        return True

    return False

def do_intersect(p1:Point, q1:Point, p2:Point, q2:Point) -> bool:
    """
    Determines if line segment 'p1q1' and 'p2q2' intersect.
    """
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # # Special Cases
    # # p1, q1 and p2 are collinear and p2 lies on segment p1q1
    # if o1 == 0 and on_segment(p1, p2, q1):
    #     return True

    # # p1, q1 and q2 are collinear and q2 lies on segment p1q1
    # if o2 == 0 and on_segment(p1, q2, q1):
    #     return True

    # # p2, q2 and p1 are collinear and p1 lies on segment p2q2
    # if o3 == 0 and on_segment(p2, p1, q2):
    #     return True

    # # p2, q2 and q1 are collinear and q1 lies on segment p2q2
    # if o4 == 0 and on_segment(p2, q1, q2):
    #     return True

    return False
