import logging
from typing import Dict, Tuple

import numpy as np
import pygame
from pygame.locals import *

from .pygame_mdp import PyGameMDP
from constants import *
from model import Actor
from model import StepResult

_logger = logging.getLogger(TARGET_OPEN_MDP)

class TargetOpenMDP(PyGameMDP):
    def __init__(self, fps:int, width:int, height:int, agent_pos:Tuple[int, int], target_pos:Tuple[int, int], display:bool, trail:bool) -> None:
        super().__init__()

        self.n_state = width * height
        self.n_action = 4 # NORTH, EAST, SOUTH, WEST
        self.d_state = (width, height)

        self.fps = fps
        self.width = width
        self.height = height
        self.display = display
        self.show_trail = trail
        self.agent_start_position = agent_pos
        self.target_start_position = target_pos

        self.agent = None
        self.target = None
        self.operator = None
        self.total_episode_reward = 0
        self.values = None
        self.edge_buffer = 100
        self.agent_step_size = 1

        self.target_image = pygame.image.load("./assets/target.png")
        self.agent_image = pygame.image.load("./assets/agent.png")
        self.star_image = pygame.image.load("./assets/star.png")

        self.init_display()

    def start(self) -> np.ndarray:
        super().start()
        self.values = None
        self.total_episode_reward = 0
        self.target = Actor(-1, BLUE, self.target_start_position)
        self.agent = Actor(-1, RED, self.agent_start_position)
        state = self.get_state()
        _logger.debug(f"state:\n{state}")
        _logger.debug(f"agent position: {self.agent.get_position()}")
        _logger.debug(f"target position: {self.target.get_position()}")
        self.update_display()
        return state

    def step(self, action:int, *args) -> Tuple[float, np.ndarray, bool, Dict[str, object]]:
        self.values:np.ndarray = args[0] if len(args) > 0 else None
        self.update_agent(action)
        result = self.get_step_result()
        self.update_display()

        return result.reward, result.state, result.is_terminal, {}

    def get_state(self) -> np.ndarray:
        state = np.zeros(self.d_state, dtype=np.int32)
        state[self.target.get_position_idx()] = 1
        state[self.agent.get_position_idx()] = 1
        return state

    def update_display(self) -> None:
        if self.display:
            self.is_quit()
            self.surface.fill((213, 216, 220))
            self.draw_trail()
            self.draw_target()
            self.draw_agent()
            pygame.display.update()
            self.game_clock.tick(self.fps)

    def draw_trail(self) -> None:
        if self.show_trail and len(self.agent.position_history) > 1:
            for i in range(1, len(self.agent.position_history)):
                line_start_pos = self.agent.position_history[i - 1]
                line_end_pos = self.agent.position_history[i]
                pygame.draw.line(self.surface, RED, line_start_pos, line_end_pos, width=2)

    def draw_target(self) -> None:
        if self.agent.get_position() != self.target.get_position():
            rect = self.target_image.get_rect()
            rect.center = self.target.get_position()
            self.surface.blit(self.target_image, rect)

    def draw_agent(self) -> None:
        if self.agent.get_position() == self.target.get_position():
            rect = self.star_image.get_rect()
            rect.center = self.agent.get_position()
            self.surface.blit(self.star_image, rect)
        else:
            rect = self.agent_image.get_rect()
            rect.center = self.agent.get_position()
            self.surface.blit(self.agent_image, rect)

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

        position = self.agent.get_position()
        if action == NORTH and position[Y] > self.edge_buffer:
            position = (position[X], position[Y] - self.agent_step_size)
            agent_moved = True
        elif action == EAST and position[X] < self.d_state[X] - self.edge_buffer:
            position = (position[X] + self.agent_step_size, position[Y])
            agent_moved = True
        elif action == SOUTH and position[Y] < self.d_state[Y] - self.edge_buffer:
            position = (position[X], position[Y] + self.agent_step_size)
            agent_moved = True
        elif action == WEST and position[X] > self.edge_buffer:
            position = (position[X] - self.agent_step_size, position[Y])
            agent_moved = True

        if position != self.agent.get_position():
            self.agent.update_position(position)

        if agent_moved:
            self.log_state_debug()

    def get_step_result(self) -> StepResult:
        state = self.get_state()

        reward = 0.

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
