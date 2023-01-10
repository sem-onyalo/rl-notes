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
from model import StepResult

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

        self.start_position = (100, self.height // 2, 90)
        self.finish_position = (self.start_position[0], self.start_position[1])
        self.finish_polygon:Rectangle = None
        self.finish_line_height = 20
        self.track:List[Point] = []
        self.track_width = 50
        self.on_track_limit = 10

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
        result = self.get_result()
        self.update_display(result.state, result)

        return result.reward, result.state, result.is_terminal, {}

    def init_display(self) -> None:
        if self.display:
            pygame.init()
            self.game_clock = pygame.time.Clock()
            self.surface = pygame.display.set_mode((self.width, self.height))
            self.font_values = pygame.font.Font(pygame.font.get_default_font(), 16)
            pygame.display.set_caption("Discrete Car MDP")

    def update_display(self, state:np.ndarray, result:StepResult=None) -> None:
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
            self.draw_debugging_text(result)
            pygame.display.update()
            self.game_clock.tick(self.fps)

    def get_result(self) -> StepResult:
        agent_polygon = self.agent.get_polygon()
        agent_viewpoint = self.agent.get_viewpoint()

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

        is_terminal = is_collision or is_finish

        state, track_points = self.get_state_and_view_track_points()

        is_on_track = False
        for point in track_points:
            is_on_track = ((
                    agent_viewpoint.x > point.x - self.on_track_limit and 
                    agent_viewpoint.x < point.x + self.on_track_limit
                ) or (
                    agent_viewpoint.y > point.y - self.on_track_limit and 
                    agent_viewpoint.y < point.y + self.on_track_limit
                )
            )
            if is_on_track:
                break

        if is_collision:
            reward = -1
        elif is_finish or is_on_track:
            reward = 1
        else:
            reward = 0

        result = StepResult()
        result.is_terminal = is_terminal
        result.reward = reward
        result.state = state
        return result

    def get_state(self) -> np.ndarray:
        state, _ = self.get_state_and_view_track_points()
        return state

    def get_state_and_view_track_points(self) -> Tuple[np.ndarray, List[Point]]:
        track_points = []
        view = self.agent.get_view()
        state = np.zeros((self.agent.view_width, self.agent.view_height), dtype=np.int32)
        for point in self.track:
            is_point_in_view, relative_position = geometry.is_point_in_polygon_with_position(view, point)
            if is_point_in_view:
                row = round(relative_position.y) - 1
                col = round(relative_position.x) - 1
                if row in range(state.shape[0]) and col in range(state.shape[1]):
                    track_points.append(point)
                    state[row][col] = 1

        return state, track_points

    def build_track(self) -> None:
        self.track = []

        line_1_x = 100
        line_1_y_start = 300
        line_1_y_end = self.height - 300
        line_1 = [Point(line_1_x, y) for y in range(line_1_y_start, line_1_y_end + 1)]
        self.track += line_1

        line_2 = []
        for angle in range(180, 270 + 1):
            x = 300 + 200 * math.cos(angle * math.pi / 180)
            y = 300 + 200 * math.sin(angle * math.pi / 180)
            line_2.append(Point(x, y))
        self.track += line_2

        line_3_y = 100
        line_3_x_start = 300
        line_3_x_end = line_3_x_start + 240
        line_3 = [Point(x, line_3_y) for x in range(line_3_x_start, line_3_x_end + 1)]
        self.track += line_3

        line_4 = []
        line_4_centre = Point(540, 300)
        for angle in range(270, 360 + 1):
            x = line_4_centre.x + 200 * math.cos(angle * math.pi / 180)
            y = line_4_centre.y + 200 * math.sin(angle * math.pi / 180)
            line_4.append(Point(x, y))
        self.track += line_4

        line_5 = []
        line_5_centre = Point(940, 300)
        for angle in range(90, 180 + 1):
            x = line_5_centre.x + 200 * math.cos(angle * math.pi / 180)
            y = line_5_centre.y + 200 * math.sin(angle * math.pi / 180)
            line_5.append(Point(x, y))
        self.track += line_5

        line_6_y = 500
        line_6_x_start = 300 + 240 + 400
        line_6_x_end = self.width - 300 - 240 - 400
        line_6 = [Point(x, line_6_y) for x in range(line_6_x_start, line_6_x_end + 1)]
        self.track += line_6

        line_7 = []
        line_7_centre = Point(self.width - 300 - 240 - 400, 300)
        for angle in range(0, 90 + 1):
            x = line_7_centre.x + 200 * math.cos(angle * math.pi / 180)
            y = line_7_centre.y + 200 * math.sin(angle * math.pi / 180)
            line_7.append(Point(x, y))
        self.track += line_7

        line_8 = []
        line_8_centre = Point(self.width - 300 - 240, 300)
        for angle in range(180, 270 + 1):
            x = line_8_centre.x + 200 * math.cos(angle * math.pi / 180)
            y = line_8_centre.y + 200 * math.sin(angle * math.pi / 180)
            line_8.append(Point(x, y))
        self.track += line_8

        line_9_y = 100
        line_9_x_start = self.width - 300 - 240
        line_9_x_end = line_9_x_start + 240
        line_9 = [Point(x, line_9_y) for x in range(line_9_x_start, line_9_x_end + 1)]
        self.track += line_9

        line_10 = []
        line_10_centre = Point(self.width - 300, 300)
        for angle in range(270, 360 + 1):
            x = line_10_centre.x + 200 * math.cos(angle * math.pi / 180)
            y = line_10_centre.y + 200 * math.sin(angle * math.pi / 180)
            line_10.append(Point(x, y))
        self.track += line_10

        line_11 = []
        line_11_x = 1820
        line_11_y_start = 300
        line_11_y_end = self.height - 300
        line_11 = [Point(line_11_x, y) for y in range(line_11_y_start, line_11_y_end + 1)]
        self.track += line_11

        line_12 = []
        line_12_centre = Point(self.width - 300, self.height - 300)
        for angle in range(0, 90 + 1):
            x = line_12_centre.x + 200 * math.cos(angle * math.pi / 180)
            y = line_12_centre.y + 200 * math.sin(angle * math.pi / 180)
            line_12.append(Point(x, y))
        self.track += line_12

        line_13_y = self.height - 100
        line_13_x_start = 300
        line_13_x_end = self.width - 300
        line_13 = [Point(x, line_13_y) for x in range(line_13_x_start, line_13_x_end + 1)]
        self.track += line_13

        line_14 = []
        line_14_centre = Point(300, self.height - 300)
        for angle in range(90, 180 + 1):
            x = line_14_centre.x + 200 * math.cos(angle * math.pi / 180)
            y = line_14_centre.y + 200 * math.sin(angle * math.pi / 180)
            line_14.append(Point(x, y))
        self.track += line_14

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
        agent_viewpoint = self.agent.get_viewpoint()
        pygame.draw.circle(self.surface, RED, agent_viewpoint(), 2)
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

    def draw_debugging_text(self, result:StepResult) -> None:
        position = self.font_values.render(f"{self.agent.get_position()}", True, TEXT_COLOUR)
        position_rect = position.get_rect()
        position_rect.center = (self.width//2, self.height//2 - 10)
        self.surface.blit(position, position_rect)

        if result != None:
            reward = self.font_values.render(f"{result.reward}", True, TEXT_COLOUR)
            reward_rect = reward.get_rect()
            reward_rect.center = (self.width//2, self.height//2 + 10)
            self.surface.blit(reward, reward_rect)

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
