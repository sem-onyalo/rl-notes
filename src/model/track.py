import math
from typing import List
from typing import Tuple

from .point import Point
from .rectangle import Rectangle
from constants import *

class TrackPoint:
    def __init__(self, point:Point, direction:str) -> None:
        self.point = point
        self.direction = direction

class Track:
    centre:List[TrackPoint]
    outer:List[TrackPoint]
    inner:List[TrackPoint]
    finish_polygon:Rectangle

    def __init__(self, width:int, height:int, track_width:int, start_position:Tuple[int, int], finish_line_height:int) -> None:
        self.width = width
        self.height = height
        self.track_width = track_width
        self.start_position = start_position
        self.finish_line_height = finish_line_height

        self.track_details:dict[Tuple[int,int],Tuple[str,int]] = {}

    def build_track_segment_vertical(self, x:int, y_start:int, y_end:int, direction:str) -> List[Point]:
        return [TrackPoint(Point(x, y), direction) for y in range(y_start, y_end + 1)]

    def build_track_segment_horizontal(self, y:int, x_start:int, x_end:int, direction:str) -> List[Point]:
        return [TrackPoint(Point(x, y), direction) for x in range(x_start, x_end + 1)]

    def build_track_segment_curve(self, point:Point, radius:int, a_start:int, a_end:int, direction:str) -> List[Point]:
        segment = []
        for angle in range(a_start, a_end + 1):
            x = point.x + radius * math.cos(angle * math.pi / 180)
            y = point.y + radius * math.sin(angle * math.pi / 180)
            segment.append(TrackPoint(Point(x, y), direction))
        return segment

    def build_track_segment_vertical_v2(self, x:int, y_start:int, y_end:int, direction:str, section:int) -> List[Point]:
        segment = {
            (x, y): (direction, section)
            for y in range(y_start, y_end + 1)
        }
        self.track_details.update(segment)

    def build_track(self) -> None:
        self.centre = []
        self.outer = []
        self.inner = []

        padding = 100
        padding_vertex = 300
        curve_radius = 200

        self.centre += self.build_track_segment_vertical(padding, padding_vertex, self.height - padding_vertex, NORTH)
        self.outer += self.build_track_segment_vertical(padding - self.track_width, padding_vertex, self.height - padding_vertex, NORTH)
        self.inner += self.build_track_segment_vertical(padding + self.track_width, padding_vertex, self.height - padding_vertex, NORTH)

        # self.build_track_segment_vertical_v2(padding, padding_vertex, self.height - padding_vertex, NORTH, 1)
        self.build_track_segment_vertical_v2(padding - self.track_width, padding_vertex, self.height - padding_vertex, NORTH, 1)
        # self.build_track_segment_vertical_v2(padding + self.track_width, padding_vertex, self.height - padding_vertex, NORTH, 1)

        self.centre += self.build_track_segment_curve(Point(padding_vertex, padding_vertex), curve_radius, 180, 270, NORTH_EAST)
        self.outer += self.build_track_segment_curve(Point(padding_vertex, padding_vertex), curve_radius + self.track_width, 180, 270, NORTH_EAST)
        self.inner += self.build_track_segment_curve(Point(padding_vertex, padding_vertex), curve_radius - self.track_width, 180, 270, NORTH_EAST)

        self.centre += self.build_track_segment_horizontal(padding, padding_vertex, padding_vertex + 240, EAST)
        self.outer += self.build_track_segment_horizontal(padding - self.track_width, padding_vertex, padding_vertex + 240, EAST)
        self.inner += self.build_track_segment_horizontal(padding + self.track_width, padding_vertex, padding_vertex + 240, EAST)

        self.centre += self.build_track_segment_curve(Point(540, padding_vertex), curve_radius, 270, 360, SOUTH_EAST)
        self.outer += self.build_track_segment_curve(Point(540, padding_vertex), curve_radius + self.track_width, 270, 360, SOUTH_EAST)
        self.inner += self.build_track_segment_curve(Point(540, padding_vertex), curve_radius - self.track_width, 270, 360, SOUTH_EAST)

        self.centre += self.build_track_segment_curve(Point(940, padding_vertex), curve_radius, 90, 180, SOUTH_EAST)
        self.outer += self.build_track_segment_curve(Point(940, padding_vertex), curve_radius - self.track_width, 90, 180, SOUTH_EAST)
        self.inner += self.build_track_segment_curve(Point(940, padding_vertex), curve_radius + self.track_width, 90, 180, SOUTH_EAST)

        self.centre += self.build_track_segment_horizontal(500, padding_vertex + 240 + 400, self.width - padding_vertex - 240 - 400, EAST)
        self.outer += self.build_track_segment_horizontal(500 - self.track_width, padding_vertex + 240 + 400, self.width - padding_vertex - 240 - 400, EAST)
        self.inner += self.build_track_segment_horizontal(500 + self.track_width, padding_vertex + 240 + 400, self.width - padding_vertex - 240 - 400, EAST)

        self.centre += self.build_track_segment_curve(Point(self.width - padding_vertex - 240 - 400, padding_vertex), curve_radius, 0, 90, NORTH_EAST)
        self.outer += self.build_track_segment_curve(Point(self.width - padding_vertex - 240 - 400, padding_vertex), curve_radius - self.track_width, 0, 90, NORTH_EAST)
        self.inner += self.build_track_segment_curve(Point(self.width - padding_vertex - 240 - 400, padding_vertex), curve_radius + self.track_width, 0, 90, NORTH_EAST)

        self.centre += self.build_track_segment_curve(Point(self.width - padding_vertex - 240, padding_vertex), curve_radius, 180, 270, NORTH_EAST)
        self.outer += self.build_track_segment_curve(Point(self.width - padding_vertex - 240, padding_vertex), curve_radius + self.track_width, 180, 270, NORTH_EAST)
        self.inner += self.build_track_segment_curve(Point(self.width - padding_vertex - 240, padding_vertex), curve_radius - self.track_width, 180, 270, NORTH_EAST)

        self.centre += self.build_track_segment_horizontal(padding, self.width - padding_vertex - 240, self.width - padding_vertex, EAST)
        self.outer += self.build_track_segment_horizontal(padding - self.track_width, self.width - padding_vertex - 240, self.width - padding_vertex, EAST)
        self.inner += self.build_track_segment_horizontal(padding + self.track_width, self.width - padding_vertex - 240, self.width - padding_vertex, EAST)

        self.centre += self.build_track_segment_curve(Point(self.width - padding_vertex, padding_vertex), curve_radius, 270, 360, SOUTH_EAST)
        self.outer += self.build_track_segment_curve(Point(self.width - padding_vertex, padding_vertex), curve_radius + self.track_width, 270, 360, SOUTH_EAST)
        self.inner += self.build_track_segment_curve(Point(self.width - padding_vertex, padding_vertex), curve_radius - self.track_width, 270, 360, SOUTH_EAST)

        self.centre += self.build_track_segment_vertical(self.width - padding, padding_vertex, self.height - padding_vertex, SOUTH)
        self.outer += self.build_track_segment_vertical(self.width - padding - self.track_width, padding_vertex, self.height - padding_vertex, SOUTH)
        self.inner += self.build_track_segment_vertical(self.width - padding + self.track_width, padding_vertex, self.height - padding_vertex, SOUTH)

        self.centre += self.build_track_segment_curve(Point(self.width - padding_vertex, self.height - padding_vertex), curve_radius, 0, 90, SOUTH_WEST)
        self.outer += self.build_track_segment_curve(Point(self.width - padding_vertex, self.height - padding_vertex), curve_radius - self.track_width, 0, 90, SOUTH_WEST)
        self.inner += self.build_track_segment_curve(Point(self.width - padding_vertex, self.height - padding_vertex), curve_radius + self.track_width, 0, 90, SOUTH_WEST)

        self.centre += self.build_track_segment_horizontal(self.height - padding, padding_vertex, self.width - padding_vertex, WEST)
        self.outer += self.build_track_segment_horizontal(self.height - padding - self.track_width, padding_vertex, self.width - padding_vertex, WEST)
        self.inner += self.build_track_segment_horizontal(self.height - padding + self.track_width, padding_vertex, self.width - padding_vertex, WEST)

        self.centre += self.build_track_segment_curve(Point(padding_vertex, self.height - padding_vertex), curve_radius, 90, 180, NORTH_WEST)
        self.outer += self.build_track_segment_curve(Point(padding_vertex, self.height - padding_vertex), curve_radius - self.track_width, 90, 180, NORTH_WEST)
        self.inner += self.build_track_segment_curve(Point(padding_vertex, self.height - padding_vertex), curve_radius + self.track_width, 90, 180, NORTH_WEST)

        finish_topleft      = Point(self.start_position[X] - self.track_width, self.start_position[Y] - (self.finish_line_height // 2))
        finish_topright     = Point(self.start_position[X] + self.track_width, self.start_position[Y] - (self.finish_line_height // 2))
        finish_bottomleft   = Point(self.start_position[X] - self.track_width, self.start_position[Y] + (self.finish_line_height // 2))
        finish_bottomright  = Point(self.start_position[X] + self.track_width, self.start_position[Y] + (self.finish_line_height // 2))
        self.finish_polygon = Rectangle(finish_topleft, finish_topright, finish_bottomright, finish_bottomleft)
