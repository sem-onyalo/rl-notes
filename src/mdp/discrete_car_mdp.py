import logging
import math
import sys
import time
from typing import Dict, Tuple

import numpy as np
import pygame
from pygame.locals import *

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

class Agent:
    width:int = 96 # width of ./assets/car.png
    height:int = 44 # height of ./assets/car.png
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

    def get_p(self) -> Tuple[int, int]:
        return (self.get_x(), self.get_y())

    def get_view_p(self) -> Tuple[int, int]:
        radius = self.width // 2
        return (self._get_rotation_x(radius, self.get_a()), self._get_rotation_y(radius, self.get_a()))

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

    def _get_rotation_x(self, radius:float, angle:float) -> float:
        return round(self.get_x() + radius * math.cos(angle * math.pi / 180), self.precision)

    def _get_rotation_y(self, radius:float, angle:float) -> float:
        return round(self.get_y() - radius * math.sin(angle * math.pi / 180), self.precision)

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
            self.draw_debugging_text()
            pygame.display.update()
            self.game_clock.tick(self.fps)

    def draw_track(self) -> None:
        position = (self.width // 2, self.height // 2)
        pygame.draw.circle(self.surface, (235, 237, 239), position, self.rad, 2)

    def draw_agent(self) -> None:
        car_view = self.agent.get_view_p()

        radius = round(math.sqrt((self.agent.width//2)**2 + (self.agent.height//2)**2))

        angle_topright = math.atan2(self.agent.height//2, self.agent.width//2)
        angle_topleft = -angle_topright + math.pi
        angle_bottomleft = angle_topright + math.pi
        angle_bottomright = -angle_topright

        topright_x = self.agent._get_rotation_x(radius, self.agent.get_a() + (angle_topright * 180 / math.pi))
        topright_y = self.agent._get_rotation_y(radius, self.agent.get_a() + (angle_topright * 180 / math.pi))

        topleft_x = self.agent._get_rotation_x(radius, self.agent.get_a() + (angle_topleft * 180 / math.pi))
        topleft_y = self.agent._get_rotation_y(radius, self.agent.get_a() + (angle_topleft * 180 / math.pi))

        bottomleft_x = self.agent._get_rotation_x(radius, self.agent.get_a() + (angle_bottomleft * 180 / math.pi))
        bottomleft_y = self.agent._get_rotation_y(radius, self.agent.get_a() + (angle_bottomleft * 180 / math.pi))

        bottomright_x = self.agent._get_rotation_x(radius, self.agent.get_a() + (angle_bottomright * 180 / math.pi))
        bottomright_y = self.agent._get_rotation_y(radius, self.agent.get_a() + (angle_bottomright * 180 / math.pi))

        pygame.draw.circle(self.surface, BLUE, self.agent.get_p(), 2)
        pygame.draw.circle(self.surface, BLUE, (topright_x, topright_y), 2)
        pygame.draw.circle(self.surface, BLUE, (topleft_x, topleft_y), 2)
        pygame.draw.circle(self.surface, BLUE, (bottomleft_x, bottomleft_y), 2)
        pygame.draw.circle(self.surface, BLUE, (bottomright_x, bottomright_y), 2)
        pygame.draw.polygon(self.surface, BLUE, [(topright_x, topright_y), (topleft_x, topleft_y), (bottomleft_x, bottomleft_y), (bottomright_x, bottomright_y)], 1)

        pygame.draw.circle(self.surface, RED, car_view, 2)

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
                elif pressed[K_LEFT]:
                    action = LEFT
                elif pressed[K_RIGHT]:
                    action = RIGHT

        if action == LEFT:
            self.agent.turn_left()
        elif action == RIGHT:
            self.agent.turn_right()

        self.agent.forward()

    def draw_debugging_text(self) -> None:
        pos = self.font_values.render(f"{self.agent.get_position()}", True, TEXT_COLOUR)
        pos_rect = pos.get_rect()
        pos_rect.center = (self.width//2, self.height//2 - pos_rect.size[Y] - 5)
        self.surface.blit(pos, pos_rect)

        view = self.font_values.render(f"{self.agent.get_view_p()}", True, TEXT_COLOUR)
        view_rect = view.get_rect()
        view_rect.center = (self.width//2, self.height//2 + view_rect.size[Y] + 5)
        self.surface.blit(view, view_rect)
