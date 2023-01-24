import logging
import sys
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pygame
from pygame.locals import *

from .discrete_car import Agent
from .pygame_mdp import PyGameMDP
from .pygame_mdp import TEXT_COLOUR
from constants import *
from model import geometry
from model import Point
from model import Rectangle
from model import StepResult
from model import Track

FORWARD = 0
LEFT = 1
RIGHT = 2

_logger = logging.getLogger(DISCRETE_CAR_MDP)

class DiscreteCarMDP(PyGameMDP):
    def __init__(self, *args) -> None:
        super().__init__()

        self.rad = args[0]
        self.fps = args[1]
        self.width = args[2]
        self.height = args[3]
        self.display = args[4]
        self.show_trail = args[5]

        self.operator = None
        self.total_episode_reward = 0

        self.start_position = (100, self.height // 2, 90)
        self.finish_position = (self.start_position[0], self.start_position[1])
        self.finish_line_height = 20
        self.track_width = 50
        self.on_track_limit = 10

        self.agent = Agent(self.start_position)
        self.track = Track(self.width, self.height, self.track_width, self.start_position, self.finish_line_height)
        self.track.build_track()

        self.n_state = self.agent.view_width * self.agent.view_height
        self.n_action = 3 # FORWARD, LEFT, RIGHT
        self.d_state = (self.agent.view_height, self.agent.view_width)

        self.car_image = pygame.image.load("./assets/car.png")

        self.init_display()

    def start(self) -> np.ndarray:
        assert self.operator != None, "Set agent operator parameter before starting"
        self.agent.reset_position(self.start_position)
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

            self.surface.fill((240, 240, 240)) # self.surface.fill((123, 182, 101))
            # self.draw_trail()
            self.draw_track()
            self.draw_agent()
            self.draw_viewport(state)
            self.draw_debugging_text(result)
            pygame.display.update()
            self.game_clock.tick(self.fps)

    def get_result(self) -> StepResult:
        agent_polygon, agent_viewpoint, track_points_in_view, is_collision, state = self.get_agent_position_info()

        is_finish = (
            geometry.is_point_in_polygon(self.track.finish_polygon, agent_polygon.topright) or
            geometry.is_point_in_polygon(self.track.finish_polygon, agent_polygon.bottomright)
        )

        is_terminal = is_collision or is_finish

        is_on_track = False
        for point in track_points_in_view:
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
        _, _, _, _, state = self.get_agent_position_info()
        return state

    def get_agent_position_info(self) -> Tuple[Rectangle, Point, List[Point], bool, np.ndarray]:
        sensor = self.agent.get_sensor()
        agent_polygon = self.agent.get_polygon()
        agent_viewpoint = self.agent.get_viewpoint()

        is_collision = False
        track_points_in_view = []
        state = np.zeros((self.agent.view_width, self.agent.view_height), dtype=np.int32)
        for i in range(len(self.track.centre)):
            is_point_in_view, relative_position = geometry.is_point_in_polygon_with_position(sensor.front, self.track.centre[i].point)
            if is_point_in_view:
                row = round(relative_position.y) - 1
                col = round(relative_position.x) - 1
                if row in range(state.shape[0]) and col in range(state.shape[1]):
                    track_points_in_view.append(self.track.centre[i].point)
                    state[row][col] = 1

        # track_points_in_sensor = []
        # for point in sensor.left.points:
        #     if point() in self.track.track_details:
        #         track_points_in_sensor.append(point)
        # distance = sensor.left.get_distance(track_points_in_sensor) if len(track_points_in_sensor) > 0 else None

        return agent_polygon, agent_viewpoint, track_points_in_view, is_collision, state

    def draw_track(self) -> None:
        for i in range(len(self.track.centre)):
            pygame.draw.circle(self.surface, (64, 64, 64), self.track.centre[i].point(), self.track_width)
            # pygame.draw.circle(self.surface, (235, 237, 239), self.track.outer[i].point(), 1)
            # pygame.draw.circle(self.surface, (235, 237, 239), self.track.inner[i].point(), 1)
            pygame.draw.circle(self.surface, GREEN, self.track.outer[i].point(), 1)

        for p in self.track.track_details:
            pygame.draw.circle(self.surface, BLACK, p, 3)

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
        if True:
            sensor = self.agent.get_sensor()
            self.draw_rectangle(sensor.front, RED)
            self.draw_points(sensor.left.points, GREEN)
            self.draw_rectangle(sensor.right, GREEN)

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
                  pygame.draw.circle(self.surface, (235, 237, 239), (viewport_point_x, viewport_point_y), 1)

    def draw_rectangle(self, rectangle:Rectangle, color:Tuple[int, int, int]) -> None:
        assert isinstance(rectangle, Rectangle), f"rectangle must be of type {Rectangle}, not {type(rectangle)}"
        pygame.draw.polygon(self.surface, color, [rectangle.topright(), rectangle.topleft(), rectangle.bottomleft(), rectangle.bottomright()], 1)

    def draw_points(self, points:List[Point], color:Tuple[int, int, int], size:int=1) -> None:
        for point in points:
            pygame.draw.circle(self.surface, color, point(), size)

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

        if self.operator == HUMAN:
            if action_test == FORWARD:
                self.agent.forward()
                self.log_state_debug()
        else:
            self.agent.forward()

    def draw_debugging_text(self, result:StepResult) -> None:
        sensor = self.agent.get_sensor()
        track_points_in_sensor = []
        for point in sensor.left.points:
            if point() in self.track.track_details:
                track_points_in_sensor.append(point)
        distance = sensor.left.get_distance(track_points_in_sensor) if len(track_points_in_sensor) > 0 else None
        pygame.draw.circle(self.surface, RED, sensor.left.anchor_point(), 5)

        position_text = self.font_values.render(f"{self.agent.get_position()}", True, TEXT_COLOUR)
        position_rect = position_text.get_rect()
        position_rect.center = (self.width//2, self.height//2 + 60)
        self.surface.blit(position_text, position_rect)

        if result != None:
            reward_text = self.font_values.render(f"{result.reward}", True, TEXT_COLOUR)
            reward_rect = reward_text.get_rect()
            reward_rect.center = (self.width//2, self.height//2 + 80)
            self.surface.blit(reward_text, reward_rect)

        d = distance if distance != None else 0
        distance_text = self.font_values.render(f"{d:.1f}", True, TEXT_COLOUR)
        distance_rect = distance_text.get_rect()
        distance_rect.center = (self.width//2 - 20, self.height//2 + 100)
        self.surface.blit(distance_text, distance_rect)

    def log_state_debug(self) -> None:
        sensor = self.agent.get_sensor()
        state = np.zeros((self.agent.view_width, self.agent.view_height), dtype=np.int32)
        points_in_view_count = 0
        for track_point in self.track.centre:
            is_point_in_view, relative_position = geometry.is_point_in_polygon_with_position(sensor.front, track_point.point)
            if is_point_in_view:
                row = round(relative_position.y) - 1
                col = round(relative_position.x) - 1
                if row in range(state.shape[0]) and col in range(state.shape[1]):
                    _logger.debug(f"{track_point.point()}, {relative_position()}, {sensor.front.topright()}, {sensor.front.topleft()}")
                    state[row][col] = 1
                    points_in_view_count += 1
        _logger.debug(f"state:\n{state}")
        _logger.debug(f"points_in_view_count: {points_in_view_count}")
        pass
