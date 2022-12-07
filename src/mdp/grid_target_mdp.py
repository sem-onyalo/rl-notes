import logging
import time
from typing import Dict, Tuple

import numpy as np
import pygame
import sys
from pygame.locals import *

from .mdp import MDP
from constants import *

X = 0
Y = 1
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

    def get_previous_position(self) -> Tuple[int, int]:
        return self.position_history[-1]

    def get_position_idx(self) -> Tuple[int, int]:
        return (self.get_x() - 1, self.get_y() - 1)

    def get_x(self) -> int:
        return self.position[0]

    def get_y(self) -> int:
        return self.position[1]

    def update_position(self, position:Tuple[int, int]) -> None:
        self.position = position
        self.position_history.append(self.position)

class GridTargetMDP(MDP):
    n_actions:int = 4 # NORTH, EAST, SOUTH, WEST
    def __init__(self, dim:int, fps:int, width:int, height:int, agent_pos:Tuple[int, int], target_pos:Tuple[int, int], display:bool, trail:bool) -> None:
        super().__init__()

        self.dim = dim
        self.fps = fps
        self.width = width
        self.height = height
        self.display = display
        self.show_trail = trail
        self.n_states = dim ** 2
        self.agent_start_position = agent_pos
        self.target_start_position = target_pos
        self.block_size = (round(self.width/self.dim), round(self.height/self.dim))

        self.agent = None
        self.target = None
        self.operator = None
        self.debounce_val = 100
        self.debounce = time.time_ns()
        self.total_episode_reward = 0

        self.init_display()

    def start(self) -> np.ndarray:
        assert self.operator != None, "Set agent operator parameter before starting"
        self.target = self.build_actor(self.target_start_position, BLUE)
        self.agent = self.build_actor(self.agent_start_position, RED)
        self.total_episode_reward = 0
        self.update_display(False)
        state = self.get_state()
        _logger.debug(f"state:\n{state}")
        return state

    def step(self, action:int) -> Tuple[float, np.ndarray, bool, Dict[str, object]]:
        agent_moved = self.update_agent(action)
        reward = self.update_reward(agent_moved)
        is_terminal = self.get_is_terminal()

        state = self.get_state()
        if agent_moved:
            _logger.debug(f"state:\n{state}")

        self.update_display(is_terminal)

        return reward, state, is_terminal, {}

    def init_display(self) -> None:
        if self.display:
            pygame.init()
            self.game_clock = pygame.time.Clock()
            self.surface = pygame.display.set_mode((self.width, self.height))
            self.font = pygame.font.Font(pygame.font.get_default_font(), 64)
            pygame.display.set_caption("Grid Target MDP")

    def update_display(self, is_terminal:bool) -> None:
        if self.display:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

            self.surface.fill((213, 216, 220))
            self.draw_grid_lines_x((235, 237, 239))
            self.draw_grid_lines_y((235, 237, 239))
            self.draw_trail()
            self.draw_target()
            self.draw_agent()
            if not is_terminal:
                self.draw_reward()
            else:
                self.draw_finish()
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
        pygame.draw.circle(self.surface, self.target.colour, pos, self.target.size)

    def draw_agent(self) -> None:
        pos = self.get_display_position(self.agent.get_position())
        pygame.draw.circle(self.surface, RED, pos, self.agent.size)

    def draw_reward(self) -> None:
        text = self.font.render(f"{int(self.total_episode_reward)}", True, (36, 113, 163))
        text_rect = text.get_rect()
        text_rect.center = (self.width // 2, self.height // 2)
        self.surface.blit(text, text_rect)

    def draw_finish(self) -> None:
        text = self.font.render(f"DONE!", True, (36, 113, 163))
        text_rect = text.get_rect()
        text_rect.center = (self.width // 2, self.height // 2)
        self.surface.blit(text, text_rect)

    def build_actor(self, position:Tuple[int, int], colour:Tuple[int, int, int]) -> Actor:
        radius = round((self.block_size[0] if self.block_size[0] <= self.block_size[1] else self.block_size[1]) / 2) - 4
        return Actor(radius, colour, position)

    def get_display_position(self, position:Tuple[int, int]) -> Tuple[int, int]:
        x = round(self.width / self.dim) * position[X] - round(self.block_size[0] / 2)
        y = round(self.height / self.dim) * position[Y] - round(self.block_size[1] / 2)
        return x, y

    def update_agent(self, action:int) -> bool:
        if self.operator == HUMAN:
            action = -1
            if self.check_input():
                pressed = pygame.key.get_pressed()
                if pressed[K_UP]:
                    action = NORTH
                elif pressed[K_RIGHT]:
                    action = EAST
                elif pressed[K_DOWN]:
                    action = SOUTH
                elif pressed[K_LEFT]:
                    action = WEST

        position = self.agent.get_position()
        if action == NORTH and position[Y] > 1:
            position = (position[X], position[Y] - 1)
        elif action == EAST and position[X] < self.dim:
            position = (position[X] + 1, position[Y])
        elif action == SOUTH and position[Y] < self.dim:
            position = (position[X], position[Y] + 1)
        elif action == WEST and position[X] > 1:
            position = (position[X] - 1, position[Y])

        agent_moved = False
        if position != self.agent.get_position():
            self.agent.update_position(position)
            agent_moved = True

        return agent_moved

    def check_input(self) -> bool:
        if ((time.time_ns() - self.debounce) / 1e6) < self.debounce_val:
            return False
        else:
            self.debounce = time.time_ns()
            return True

    def update_reward(self, agent_moved:bool) -> float:
        reward = 0.
        if agent_moved and self.agent.get_position() != self.target.get_position():
            reward = -1
        self.total_episode_reward += reward
        return reward

    def get_is_terminal(self) -> bool:
        return self.agent.get_position() == self.target.get_position()

    def get_state(self) -> np.ndarray:
        _logger.debug(f"agent: ({self.agent.get_position()})")
        _logger.debug(f"target: ({self.target.get_position()})")
        state = np.zeros((self.dim, self.dim), dtype=np.int32)
        state[self.agent.get_position_idx()] = 1
        state[self.target.get_position_idx()] = 1
        return state

    def set_operator(self, operator:str) -> None:
        self.operator = operator
        if self.operator == HUMAN:
            self.display = True
            self.init_display()
