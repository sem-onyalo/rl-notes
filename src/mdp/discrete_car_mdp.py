import logging
import sys
import time
from typing import Dict, Tuple

import numpy as np
import pygame
from pygame.locals import *

from .pygame_mdp import PyGameMDP
from .pygame_mdp import X
from .pygame_mdp import Y
from constants import *

FORWARD = 0
LEFT = 1
RIGHT = 2

_logger = logging.getLogger(DISCRETE_CAR_MDP)

class Agent:
    def __init__(self, position) -> None:
        self.position = position
        self.position_history = []
        self.position_history.append(position)

    def get_position(self) -> Tuple[int, int, int]:
        return self.position

    def update_position(self, position:Tuple[int, int, int]) -> None:
        self.position = position
        self.position_history.append(self.position)

    def get_x(self) -> int:
        return self.position[0]

    def get_y(self) -> int:
        return self.position[1]

    def get_a(self) -> int:
        return self.position[2]

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

        self.car_image = pygame.image.load("./assets/car.png")

        self.init_display()

    def start(self) -> np.ndarray:
        assert self.operator != None, "Set agent operator parameter before starting"
        self.agent = self.build_agent()
        self.update_display()

    def step(self, action:int) -> Tuple[float, np.ndarray, bool, Dict[str, object]]:
        self.update_agent(action)
        self.update_display()

    def init_display(self) -> None:
        if self.display:
            pygame.init()
            self.game_clock = pygame.time.Clock()
            self.surface = pygame.display.set_mode((self.width, self.height))
            self.font_values = pygame.font.Font(pygame.font.get_default_font(), 16)
            pygame.display.set_caption("Discrete Car MDP")

    def update_display(self) -> None:
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
            pygame.display.update()
            self.game_clock.tick(self.fps)

    def draw_track(self) -> None:
        position = (self.width // 2, self.height // 2)
        pygame.draw.circle(self.surface, (235, 237, 239), position, self.rad, 2)

    def draw_agent(self) -> None:
        rect = self.car_image.get_rect()
        rect.center = (self.agent.get_x(), self.agent.get_y())
        self.surface.blit(pygame.transform.rotate(self.car_image, self.agent.get_a()), rect)

    def build_agent(self) -> Agent:
        position = ((self.width // 2) + self.rad, self.height // 2, 0)
        agent = Agent(position)
        return agent

    def update_agent(self, action:int) -> None:
        if self.operator == HUMAN:
            action = -1
            if self.check_input():
                pressed = pygame.key.get_pressed()
                if pressed[K_UP]:
                    action = FORWARD
                elif pressed[K_RIGHT]:
                    action = RIGHT
                elif pressed[K_LEFT]:
                    action = LEFT

        position = self.agent.get_position()
        if action == FORWARD:
            pass
        elif action == RIGHT:
            position = (self.agent.get_x(), self.agent.get_y(), self.agent.get_a() - 5)
        elif action == LEFT:
            position = (self.agent.get_x(), self.agent.get_y(), self.agent.get_a() + 5)

        if position != self.agent.get_position():
            self.agent.update_position(position)
