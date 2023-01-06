import logging
import math
import time
from typing import Dict, Tuple

import numpy as np
import pygame
import sys
from pygame.locals import *

from .pygame_mdp import PyGameMDP
from .pygame_mdp import TEXT_COLOUR
from .pygame_mdp import X
from .pygame_mdp import Y
from constants import *
# from function import Policy

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

_logger = logging.getLogger(GRID_TARGET_MDP)

class Actor:
    def __init__(self, size, colour, position) -> None:
        self.size = size
        self.colour = colour
        self.position = position
        self.position_history = []
        self.position_history.append(position)

    def get_position(self) -> Tuple[int, int]:
        return self.position

    def update_position(self, position:Tuple[int, int]) -> None:
        self.position = position
        self.position_history.append(self.position)

    def get_x(self) -> int:
        return self.position[0]

    def get_y(self) -> int:
        return self.position[1]

    def get_position_idx(self) -> Tuple[int, int]:
        return (self.get_x() - 1, self.get_y() - 1)

class GridTargetMDP(PyGameMDP):
    def __init__(self, dim:int, fps:int, width:int, height:int, agent_pos:Tuple[int, int], target_pos:Tuple[int, int], display:bool, trail:bool) -> None:
        super().__init__()

        self.n_state = dim ** 2
        self.n_action = 4 # NORTH, EAST, SOUTH, WEST
        self.d_state = (dim, dim)

        self.dim = dim
        self.fps = fps
        self.width = width
        self.height = height
        self.display = display
        self.show_trail = trail
        self.agent_start_position = agent_pos
        self.target_start_position = target_pos
        self.cell_size = (round(self.width/self.dim), round(self.height/self.dim))

        self.agent = None
        self.target = None
        self.operator = None
        self.total_episode_reward = 0

        self.target_image = pygame.image.load("./assets/target.png")
        self.agent_image = pygame.image.load("./assets/agent.png")
        self.star_image = pygame.image.load("./assets/star.png")

        self.init_display()

    def start(self) -> np.ndarray:
        assert self.operator != None, "Set agent operator parameter before starting"
        self.total_episode_reward = 0
        self.target = self.build_actor(self.target_start_position, BLUE)
        self.agent = self.build_actor(self.agent_start_position, RED)
        state = self.get_state()
        _logger.debug(f"state:\n{state}")
        self.update_display()
        return state

    def step(self, action:int) -> Tuple[float, np.ndarray, bool, Dict[str, object]]:
        agent_moved = self.update_agent(action)
        reward = self.update_reward()
        is_terminal = self.get_is_terminal()

        state = self.get_state()
        if agent_moved:
            _logger.debug(f"state:\n{state}")

        self.update_display()

        return reward, state, is_terminal, {}

    def init_display(self) -> None:
        if self.display:
            pygame.init()
            self.game_clock = pygame.time.Clock()
            self.surface = pygame.display.set_mode((self.width, self.height))
            self.font_values = pygame.font.Font(pygame.font.get_default_font(), 16)
            pygame.display.set_caption("Grid Target MDP")

    def set_policy(self, policy) -> None:
        self.policy = policy

    def update_display(self) -> None:
        if self.display:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

            self.surface.fill((213, 216, 220))
            self.draw_grid_lines_x((235, 237, 239))
            self.draw_grid_lines_y((235, 237, 239))
            self.draw_trail()
            self.draw_values()
            self.draw_target()
            self.draw_agent()
            pygame.display.update()
            self.game_clock.tick(self.fps)

    def draw_grid_lines_x(self, colour:Tuple[int, int, int]) -> None:
        buffer = 0
        x = round(self.width/self.dim)
        for _ in range(0, self.dim):
            start_pos = (buffer + x, 0)
            end_pos = (buffer + x, self.height)
            pygame.draw.line(self.surface, colour, start_pos, end_pos, width=2)
            buffer += x

    def draw_grid_lines_y(self, colour:Tuple[int, int, int]) -> None:
        buffer = 0
        y = round(self.height/self.dim)
        for _ in range(0, self.dim):
            start_pos = (0, buffer + y)
            end_pos = (self.width, buffer + y)
            pygame.draw.line(self.surface, colour, start_pos, end_pos, width=2)
            buffer += y

    def draw_trail(self) -> None:
        if self.show_trail and len(self.agent.position_history) > 1:
            for i in range(1, len(self.agent.position_history)):
                line_start_pos = self.get_display_position(self.agent.position_history[i - 1])
                line_end_pos = self.get_display_position(self.agent.position_history[i])
                pygame.draw.line(self.surface, RED, line_start_pos, line_end_pos, width=2)

    def draw_target(self) -> None:
        pos = self.get_display_position(self.target.get_position())
        if self.agent.get_position() != self.target.get_position():
            rect = self.target_image.get_rect()
            rect.center = pos
            self.surface.blit(self.target_image, rect)

    def draw_agent(self) -> None:
        pos = self.get_display_position(self.agent.get_position())
        if self.agent.get_position() == self.target.get_position():
            rect = self.star_image.get_rect()
            rect.center = pos
            self.surface.blit(self.star_image, rect)
        else:
            rect = self.agent_image.get_rect()
            rect.center = pos
            self.surface.blit(self.agent_image, rect)

    def draw_values(self) -> None:
        if self.operator != HUMAN:
            for x in range(0, self.dim):
                for y in range(0, self.dim):
                    state = self.get_state_with_actor_position((x,y))
                    values = self.policy.get_values(state)
                    max_idx = values.argmax()
                    text_values = []

                    text_north = self.font_values.render(f"{values[NORTH]:.2f}", True, TEXT_COLOUR)
                    text_x = (self.cell_size[X] * x) + (self.cell_size[X] // 2)
                    text_y = (self.cell_size[Y] * y) + math.floor(self.cell_size[Y] * .1)
                    text_north_rect = text_north.get_rect()
                    text_north_rect.center = (text_x, text_y)
                    text_values.append((text_north, text_north_rect))

                    text_east = self.font_values.render(f"{values[EAST]:.2f}", True, TEXT_COLOUR)
                    text_x = (self.cell_size[X] * x) + math.floor(self.cell_size[X] * .9)
                    text_y = (self.cell_size[Y] * y) + (self.cell_size[Y] // 2)
                    text_east_rect = text_east.get_rect()
                    text_east_rect.center = (text_x, text_y)
                    text_values.append((text_east, text_east_rect))

                    text_south = self.font_values.render(f"{values[SOUTH]:.2f}", True, TEXT_COLOUR)
                    text_x = (self.cell_size[X] * x) + (self.cell_size[X] // 2)
                    text_y = (self.cell_size[Y] * y) + math.floor(self.cell_size[Y] * .9)
                    text_south_rect = text_south.get_rect()
                    text_south_rect.center = (text_x, text_y)
                    text_values.append((text_south, text_south_rect))

                    text_west = self.font_values.render(f"{values[WEST]:.2f}", True, TEXT_COLOUR)
                    text_x = (self.cell_size[X] * x) + math.floor(self.cell_size[X] * .1)
                    text_y = (self.cell_size[Y] * y) + (self.cell_size[Y] // 2)
                    text_west_rect = text_west.get_rect()
                    text_west_rect.center = (text_x, text_y)
                    text_values.append((text_west, text_west_rect))
                    
                    if not all([v == 0 for v in values]):
                        topleft = (text_values[max_idx][1].topleft[0] - 2, text_values[max_idx][1].topleft[1] - 2)
                        size = (text_values[max_idx][1].size[X] + 4, text_values[max_idx][1].size[Y] + 4)
                        pygame.draw.rect(self.surface, RED, pygame.Rect(topleft, size))

                    self.surface.blit(text_values[NORTH][0], text_values[NORTH][1])
                    self.surface.blit(text_values[EAST][0], text_values[EAST][1])
                    self.surface.blit(text_values[SOUTH][0], text_values[SOUTH][1])
                    self.surface.blit(text_values[WEST][0], text_values[WEST][1])

    def build_actor(self, position:Tuple[int, int], colour:Tuple[int, int, int]) -> Actor:
        radius = round((self.cell_size[0] if self.cell_size[0] <= self.cell_size[1] else self.cell_size[1]) / 2) - 4
        return Actor(radius, colour, position)

    def get_display_position(self, position:Tuple[int, int]) -> Tuple[int, int]:
        x = self.cell_size[X] * position[X] - (self.cell_size[X] // 2)
        y = self.cell_size[Y] * position[Y] - (self.cell_size[Y] // 2)
        return x, y

    def update_agent(self, action:int) -> bool:
        agent_moved = False
        if self.operator == HUMAN:
            action = -1
            if self.check_input():
                pressed = pygame.key.get_pressed()
                if pressed[K_UP]:
                    action = NORTH
                    agent_moved = True
                elif pressed[K_RIGHT]:
                    action = EAST
                    agent_moved = True
                elif pressed[K_DOWN]:
                    action = SOUTH
                    agent_moved = True
                elif pressed[K_LEFT]:
                    action = WEST
                    agent_moved = True
        else:
            agent_moved = True

        position = self.agent.get_position()
        if action == NORTH and position[Y] > 1:
            position = (position[X], position[Y] - 1)
        elif action == EAST and position[X] < self.dim:
            position = (position[X] + 1, position[Y])
        elif action == SOUTH and position[Y] < self.dim:
            position = (position[X], position[Y] + 1)
        elif action == WEST and position[X] > 1:
            position = (position[X] - 1, position[Y])

        if position != self.agent.get_position():
            self.agent.update_position(position)

        return agent_moved

    def update_reward(self) -> float:
        # reward = 1. if self.agent.get_position() == self.target.get_position() else 0.
        # reward = 0. if self.agent.get_position() == self.target.get_position() else -1.
        reward = 1. if self.agent.get_position() == self.target.get_position() else -1.
        self.total_episode_reward += reward
        return reward

    def get_is_terminal(self) -> bool:
        return self.agent.get_position() == self.target.get_position()

    def get_state(self) -> np.ndarray:
        _logger.debug(f"agent: ({self.agent.get_position()})")
        _logger.debug(f"target: ({self.target.get_position()})")
        return self.get_state_with_actor_position(self.agent.get_position_idx())

    def get_state_with_actor_position(self, agent_position:Tuple[int, int]) -> np.ndarray:
        state = np.zeros((self.dim, self.dim), dtype=np.int32)
        state[self.target.get_position_idx()] = 1
        state[agent_position] = 1
        return state
