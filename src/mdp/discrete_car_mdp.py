import logging
import math
import sys
from typing import Dict, List, Tuple

import numpy as np
import pygame
from pygame.locals import *

from .discrete_car import Agent
from .pygame_mdp import PyGameMDP
from .pygame_mdp import TEXT_COLOUR
from .pygame_mdp import X
from .pygame_mdp import Y
from .pygame_mdp import A
from constants import *
from model import geometry
from model import Point
from model import Rectangle

FORWARD = 0
LEFT = 1
RIGHT = 2

_logger = logging.getLogger(DISCRETE_CAR_MDP)

class DiscreteCarMDP(PyGameMDP):
    def __init__(self, *args) -> None:
        super().__init__()

        self.n_state = None
        self.n_action = 3 # FORWARD, LEFT, RIGHT
        self.d_state = None

        self.rad = args[0]
        self.fps = args[1]
        self.width = args[2]
        self.height = args[3]
        self.display = args[4]
        self.show_trail = args[5]

        self.agent = None
        self.operator = None
        self.total_episode_reward = 0

        self.start_position = (self.width // 2 + self.rad, self.height // 2, 90)
        self.finish_position = (self.start_position[0], self.start_position[1])
        self.finish_polygon:Rectangle = None
        self.finish_line_height = 20
        self.track:List[Point] = []
        self.track_width = 50

        self.car_image = pygame.image.load("./assets/car.png")

        self.init_display()

    def start(self) -> np.ndarray:
        assert self.operator != None, "Set agent operator parameter before starting"
        self.build_track()
        self.agent = Agent(self.start_position)
        state = self.get_state()
        self.update_display(state)
        return state

    def step(self, action:int) -> Tuple[float, np.ndarray, bool, Dict[str, object]]:
        self.update_agent(action)
        reward = self.get_reward()
        is_terminal = self.get_is_terminal()
        state = self.get_state()
        self.update_display(state)

        return reward, state, is_terminal, {}

    def init_display(self) -> None:
        if self.display:
            pygame.init()
            self.game_clock = pygame.time.Clock()
            self.surface = pygame.display.set_mode((self.width, self.height))
            self.font_values = pygame.font.Font(pygame.font.get_default_font(), 16)
            pygame.display.set_caption("Discrete Car MDP")

    def update_display(self, state:np.ndarray) -> None:
        if self.display:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

            self.surface.fill((213, 216, 220))
            # self.draw_trail()
            # self.draw_values()
            self.draw_track()
            self.draw_agent()
            self.draw_viewport(state)
            self.draw_debugging_text()
            pygame.display.update()
            self.game_clock.tick(self.fps)

    def get_state(self) -> np.ndarray:
        state = np.zeros((self.agent.view_width, self.agent.view_height), dtype=np.int32)
        view = self.agent.get_view()
        for point in self.track:
            is_point_in_view, relative_position = geometry.is_point_in_polygon_with_position(view, point)
            if is_point_in_view:
                row = round(relative_position.y) - 1
                col = round(relative_position.x) - 1
                if row in range(state.shape[0]) and col in range(state.shape[1]):
                    state[row][col] = 1

        return state

    def build_track(self) -> None:
        self.track = [Point(self.start_position[X], y) for y in range(self.height + 1)]
        
        finish_topleft      = Point(self.start_position[X] - self.track_width, self.start_position[Y] - (self.finish_line_height // 2))
        finish_topright     = Point(self.start_position[X] + self.track_width, self.start_position[Y] - (self.finish_line_height // 2))
        finish_bottomleft   = Point(self.start_position[X] - self.track_width, self.start_position[Y] + (self.finish_line_height // 2))
        finish_bottomright  = Point(self.start_position[X] + self.track_width, self.start_position[Y] + (self.finish_line_height // 2))
        self.finish_polygon = Rectangle(finish_topleft, finish_topright, finish_bottomright, finish_bottomleft)

    def draw_track(self) -> None:
        for point in self.track:
            pygame.draw.circle(self.surface, (235, 237, 239), point(), self.track_width)

        # Finish line
        color = None
        block_width = self.finish_line_height // 2
        for y in range(self.start_position[Y] - block_width, self.start_position[Y] + block_width, 10):
            color = BLACK if color == WHITE else None
            for x in range(self.start_position[X] - self.track_width, self.start_position[X] + self.track_width, 10):
                color = WHITE if color == BLACK else BLACK
                pygame.draw.rect(self.surface, color, pygame.Rect(x, y, block_width, block_width))

    def draw_agent(self) -> None:
        agent_polygon = self.agent.get_polygon()
        # pygame.draw.circle(self.surface, BLUE, self.agent.get_centre(), 2)
        pygame.draw.circle(self.surface, BLUE, agent_polygon.topright(), 2)
        pygame.draw.circle(self.surface, BLUE, agent_polygon.topleft(), 2)
        pygame.draw.circle(self.surface, BLUE, agent_polygon.bottomleft(), 2)
        pygame.draw.circle(self.surface, BLUE, agent_polygon.bottomright(), 2)
        self.draw_rectangle(agent_polygon, BLUE)

    def draw_viewport(self, state:np.ndarray) -> None:
        # view = self.agent.get_view()
        # pygame.draw.polygon(self.surface, RED, [view.topright(), view.topleft(), view.bottomleft(), view.bottomright()], 1)

        viewport_buffer = 10
        viewport_width = self.agent.view_height
        viewport_height = self.agent.view_width
        viewport_left = self.width - viewport_buffer - viewport_width
        viewport_top = viewport_buffer
        viewport = pygame.Rect(viewport_left, viewport_top, viewport_width, viewport_height)
        pygame.draw.rect(self.surface, BLACK, viewport)
        for row in range(state.shape[0]):
            for col in range(state.shape[1]):
                if state[row][col] != 0:
                  viewport_point_x = viewport_left + col + 1
                  viewport_point_y = viewport_top + row + 1
                  pygame.draw.circle(self.surface, GREEN, (viewport_point_x, viewport_point_y), 1)

    def draw_rectangle(self, rectangle:Rectangle, color:Tuple[int, int, int]) -> None:
        assert isinstance(rectangle, Rectangle), f"rectangle must be of type {Rectangle}, not {type(rectangle)}"
        pygame.draw.polygon(self.surface, color, [rectangle.topright(), rectangle.topleft(), rectangle.bottomleft(), rectangle.bottomright()], 1)

    def update_agent(self, action:int) -> None:
        action_test = None
        if self.operator == HUMAN:
            action = -1
            if self.check_input():
                pressed = pygame.key.get_pressed()
                if pressed[K_UP]:
                    action = FORWARD
                    action_test = FORWARD
                if pressed[K_LEFT]:
                    action = LEFT
                elif pressed[K_RIGHT]:
                    action = RIGHT

        if action == LEFT:
            self.agent.turn_left()
            self.log_state_debug()
        elif action == RIGHT:
            self.agent.turn_right()
            self.log_state_debug()

        # self.agent.forward()
        if action_test == FORWARD:
            self.agent.forward()
            self.log_state_debug()

    def get_reward(self) -> float:
        pass

    def get_is_terminal(self) -> bool:
        agent_polygon = self.agent.get_polygon()

        is_collision = (
            agent_polygon.topright.x <= 0 or
            agent_polygon.topright.y <= 0 or
            agent_polygon.topright.x >= self.width or
            agent_polygon.topright.y >= self.height or

            agent_polygon.topleft.x <= 0 or
            agent_polygon.topleft.y <= 0 or
            agent_polygon.topleft.x >= self.width or
            agent_polygon.topleft.y >= self.height or

            agent_polygon.bottomleft.x <= 0 or
            agent_polygon.bottomleft.y <= 0 or
            agent_polygon.bottomleft.x >= self.width or
            agent_polygon.bottomleft.y >= self.height or

            agent_polygon.bottomright.x <= 0 or
            agent_polygon.bottomright.y <= 0 or
            agent_polygon.bottomright.x >= self.width or
            agent_polygon.bottomright.y >= self.height
        )

        is_finish = (
            geometry.is_point_in_polygon(self.finish_polygon, agent_polygon.topright) or
            geometry.is_point_in_polygon(self.finish_polygon, agent_polygon.bottomright)
        )

        return is_collision or is_finish

    # def is_point_in_polygon_with_position(self, polygon:Rectangle, point:Point) -> Tuple[bool, Point]:
    #     is_in_polygon = self.is_point_in_polygon(polygon, point)

    #     x = round(self.agent._triangle_height(point, polygon.topright, polygon.topleft), self.agent.precision)
    #     y = round(self.agent._triangle_height(point, polygon.topright, polygon.bottomright), self.agent.precision)
    #     relative_position = Point(x, y)

    #     return is_in_polygon, relative_position

    # def is_point_in_polygon(self, polygon:Rectangle, point:Point) -> bool:
    #     """
    #     Determines if the given point is inside the given view.

    #     Point is in view (i.e. point is inside polygon) if a ray from origin to point intesects the view edges an odd number of times.
    #         - Source: https://en.wikipedia.org/wiki/Point_in_polygon
    #     The ray intersects with the view edge based on the orientation of an ordered triplet of points in the plane.
    #         - Source: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

    #     Returns a tuple where element:
    #         0: true if the point is inside the view, false otherwise
    #         1: the relative position of the point inside the view
    #     """

    #     line_segments = [
    #         (polygon.topright, polygon.topleft),
    #         (polygon.topleft, polygon.bottomleft),
    #         (polygon.bottomleft, polygon.bottomright),
    #         (polygon.bottomright, polygon.topright)
    #     ]

    #     intersect_count = 0
    #     ray_start = Point(0, point.y)
    #     for line_segment in line_segments:
    #         if self.agent._do_intersect(line_segment[0], line_segment[1], ray_start, point):
    #             intersect_count += 1
    #     return intersect_count > 0 and intersect_count % 2 != 0

    def draw_debugging_text(self) -> None:
        pos = self.font_values.render(f"{self.agent.get_position()}", True, TEXT_COLOUR)
        pos_rect = pos.get_rect()
        pos_rect.center = (self.width//2, self.height//2)
        self.surface.blit(pos, pos_rect)

    def log_state_debug(self) -> None:
        state = np.zeros((self.agent.view_width, self.agent.view_height), dtype=np.int32)
        view = self.agent.get_view()
        for point in self.track:
            is_point_in_view, relative_position = geometry.is_point_in_polygon_with_position(view, point)
            if is_point_in_view:
                row = round(relative_position.y) - 1
                col = round(relative_position.x) - 1
                if row in range(state.shape[0]) and col in range(state.shape[1]):
                    _logger.debug(f"{point()}, {relative_position()}, {view.topright()}, {view.topleft()}")
                    state[row][col] = 1
        _logger.debug(f"state:\n{state}")
        pass
