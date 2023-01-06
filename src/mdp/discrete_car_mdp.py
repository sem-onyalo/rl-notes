import logging
import math
import sys
# import time
from typing import Dict, List, Tuple

import numpy as np
import pygame
from pygame.locals import *

from .discrete_car import Agent
from .discrete_car import Point
from .pygame_mdp import PyGameMDP
from .pygame_mdp import TEXT_COLOUR
from .pygame_mdp import X
from .pygame_mdp import Y
from .pygame_mdp import A
from constants import *

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
        self.track:List[Point] = []
        self.start_position = (self.width // 2 + self.rad, self.height // 2, 90)
        self.total_episode_reward = 0

        self.car_image = pygame.image.load("./assets/car.png")

        self.init_display()

    def start(self) -> np.ndarray:
        assert self.operator != None, "Set agent operator parameter before starting"
        self.build_track()
        self.agent = Agent(self.start_position)
        state = self.get_state()
        self.update_display(state)
        # _logger.debug(f"state:\n{state}")
        return state

    def step(self, action:int) -> Tuple[float, np.ndarray, bool, Dict[str, object]]:
        self.update_agent(action)
        state = self.get_state()
        self.update_display(state)
        return super().step(action)

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
            is_point_in_view, relative_position = self.agent.is_point_in_view(view, point)
            if is_point_in_view:
                row = round(relative_position.y) - 1
                col = round(relative_position.x) - 1
                if row in range(state.shape[0]) and col in range(state.shape[1]):
                    state[row][col] = 1

        return state

    def build_track(self) -> None:
        self.track = [Point(self.start_position[X], y) for y in range(self.height + 1)]

    def draw_track(self) -> None:
        for point in self.track:
            pygame.draw.circle(self.surface, (235, 237, 239), point(), 1)

    def draw_agent(self) -> None:
        agent_polygon = self.agent.get_polygon()
        pygame.draw.circle(self.surface, BLUE, self.agent.get_centre(), 2)
        pygame.draw.circle(self.surface, BLUE, agent_polygon.topright(), 2)
        pygame.draw.circle(self.surface, BLUE, agent_polygon.topleft(), 2)
        pygame.draw.circle(self.surface, BLUE, agent_polygon.bottomleft(), 2)
        pygame.draw.circle(self.surface, BLUE, agent_polygon.bottomright(), 2)
        pygame.draw.polygon(self.surface, BLUE, [agent_polygon.topright(), agent_polygon.topleft(), agent_polygon.bottomleft(), agent_polygon.bottomright()], 1)

    def draw_viewport(self, state:np.ndarray) -> None:
        view = self.agent.get_view()
        pygame.draw.polygon(self.surface, RED, [view.topright(), view.topleft(), view.bottomleft(), view.bottomright()], 1)

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

    def draw_debugging_text(self) -> None:
        pos = self.font_values.render(f"{self.agent.get_position()}", True, TEXT_COLOUR)
        pos_rect = pos.get_rect()
        pos_rect.center = (self.width//2, self.height//2)
        self.surface.blit(pos, pos_rect)

    def log_state_debug(self) -> None:
        state = np.zeros((self.agent.view_width, self.agent.view_height), dtype=np.int32)
        view = self.agent.get_view()
        for point in self.track:
            is_point_in_view, relative_position = self.agent.is_point_in_view(view, point)
            if is_point_in_view:
                row = round(relative_position.y) - 1
                col = round(relative_position.x) - 1
                if row in range(state.shape[0]) and col in range(state.shape[1]):
                    _logger.debug(f"{point()}, {relative_position()}, {view.topright()}, {view.topleft()}")
                    state[row][col] = 1
        _logger.debug(f"state:\n{state}")
        pass
