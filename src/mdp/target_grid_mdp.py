import logging
import math
from typing import Dict, Tuple

import numpy as np
import pygame
from pygame.locals import *

from .pygame_mdp import PyGameMDP
from .pygame_mdp import TEXT_COLOUR
from constants import *
from model import Actor
from model import StepResult

_logger = logging.getLogger(TARGET_GRID_MDP)

class TargetGridMDP(PyGameMDP):
    def __init__(self, grid_dim:int, fps:int, width:int, height:int, agent_pos:Tuple[int, int], target_pos:Tuple[int, int], display:bool, trail:bool) -> None:
        super().__init__()

        self.n_state = grid_dim ** 2
        self.n_action = 4 # NORTH, EAST, SOUTH, WEST
        self.d_state = (grid_dim, grid_dim)

        self.grid_dim = grid_dim
        self.fps = fps
        self.width = width
        self.height = height
        self.display = display
        self.show_trail = trail
        self.agent_start_position = agent_pos
        self.target_start_position = target_pos
        self.cell_size = (round(self.width/self.grid_dim), round(self.height/self.grid_dim))

        self.agent = None
        self.target = None
        self.operator = None
        self.total_episode_reward = 0
        self.values = None

        self.target_image = pygame.image.load("./assets/target.png")
        self.agent_image = pygame.image.load("./assets/agent.png")
        self.star_image = pygame.image.load("./assets/star.png")

        self.init_display()

    def start(self) -> np.ndarray:
        super().start()
        self.values = None
        self.total_episode_reward = 0
        self.target = self.build_actor(self.target_start_position, BLUE)
        self.agent = self.build_actor(self.agent_start_position, RED)
        state = self.get_state()
        _logger.debug(f"state:\n{state}")
        _logger.debug(f"agent position: {self.agent.get_position()} ({self.get_display_position(self.agent.get_position())})")
        _logger.debug(f"target position: {self.target.get_position()} ({self.get_display_position(self.target.get_position())})")
        self.update_display()
        return state

    def step(self, action:int, *args) -> Tuple[float, np.ndarray, bool, Dict[str, object]]:
        self.values:np.ndarray = args[0] if len(args) > 0 else None
        self.update_agent(action)
        result = self.get_step_result()
        self.update_display()

        return result.reward, result.state, result.is_terminal, {}

    def get_state(self) -> np.ndarray:
        state = np.zeros((self.grid_dim, self.grid_dim), dtype=np.int32)
        state[self.target.get_position_idx()] = 1
        state[self.agent.get_position_idx()] = 1
        return state

    def update_display(self) -> None:
        if self.display:
            self.is_quit()
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
        x = round(self.width/self.grid_dim)
        for _ in range(0, self.grid_dim):
            start_pos = (buffer + x, 0)
            end_pos = (buffer + x, self.height)
            pygame.draw.line(self.surface, colour, start_pos, end_pos, width=2)
            buffer += x

    def draw_grid_lines_y(self, colour:Tuple[int, int, int]) -> None:
        buffer = 0
        y = round(self.height/self.grid_dim)
        for _ in range(0, self.grid_dim):
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
        if self.operator != HUMAN and isinstance(self.values, np.ndarray):
            values = self.values
            for x in range(0, self.grid_dim):
                for y in range(0, self.grid_dim):
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

    def update_agent(self, action:int) -> None:
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
        elif action == EAST and position[X] < self.grid_dim:
            position = (position[X] + 1, position[Y])
        elif action == SOUTH and position[Y] < self.grid_dim:
            position = (position[X], position[Y] + 1)
        elif action == WEST and position[X] > 1:
            position = (position[X] - 1, position[Y])

        if position != self.agent.get_position():
            self.agent.update_position(position)

        if agent_moved:
            self.log_state_debug()

    def get_step_result(self) -> StepResult:
        state = self.get_state()

        # reward = 1. if self.agent.get_position() == self.target.get_position() else 0.
        # reward = 0. if self.agent.get_position() == self.target.get_position() else -1.
        reward = 1. if self.agent.get_position() == self.target.get_position() else -1.
        self.total_episode_reward += reward

        is_terminal = self.agent.get_position() == self.target.get_position()

        result = StepResult()
        result.is_terminal = is_terminal
        result.reward = reward
        result.state = state
        return result

    def log_state_debug(self) -> None:
        state = self.get_state()
        _logger.debug(f"agent: ({self.agent.get_position()})")
        _logger.debug(f"target: ({self.target.get_position()})")
        _logger.debug(f"state:\n{state}")