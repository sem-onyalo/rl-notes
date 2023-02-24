import logging
import math
import pickle
from typing import Dict, Tuple

import numpy as np
import pygame
from pygame.locals import *

from .pygame_mdp import PyGameMDP
from .pygame_mdp import TEXT_COLOUR
from constants import *
from model import Actor
from model import StepResult
from model import TrackV2

# AGENT = "./assets/agent_car.png"
# GRASS = "./assets/grass.png"
# ROAD = "./assets/road.png"
# ROAD_CURVE = "./assets/curve.road"
ROAD_LIGHT = "./assets/road_light.png"
ROAD_DARK = "./assets/road_dark.png"

_logger = logging.getLogger(RACE_TRACK_MDP)

class RaceTrackMDP(PyGameMDP):
    """
    Race track MDP based off of https://raytomely.itch.io/pseudo-3d-road-collection.
    """
    def __init__(self) -> None:
        super().__init__()

        self.fps = 60
        self.width = 640
        self.height = 480
        self.plane_center_y = self.height//2

        self.n_state = self.width * self.height
        self.n_action = 1 # FORWARD
        self.d_state = (self.width, self.height)

        self.agent_speed = 80
        self.agent_position = 0
        self.agent_position_threshold = 300  #this determine how much the road will be divided into strips
        self.road_position_speed = 4  #this determine how much the strips will stretch forward
        self.road_increment_z = 0.001

        self.track_position_y_index = -1
        self.track_position_y_increment = 10  #this is the speed at witch we traverse the curve
        self.track_curve_direction = 1
        self.track_position_x = 0
        self.track_position_x_increment = 20
        self.track_border = 300

        self.track = TrackV2(self.plane_center_y)
        self.track_view_length = len(self.track)
        self.track_value_min = min(self.track)
        self.track_value_max = max(self.track)
        self.track_value_min_scaled = 0
        self.track_value_max_scaled = 10

        # self.init_display()

    def init_display(self) -> None:
        super().init_display()
        self.road_light = pygame.image.load(ROAD_LIGHT).convert()
        self.road_dark = pygame.image.load(ROAD_DARK).convert()
        self.strip_light = pygame.Surface((self.width, 1)).convert()
        self.strip_dark = pygame.Surface((self.width, 1)).convert()
        self.strip_light.fill(self.road_light.get_at((0, 0)))
        self.strip_dark.fill(self.road_dark.get_at((0, 0)))

    def start(self) -> np.ndarray:
        super().start()
        self.log_init_debug()
        self.update_display()
        return super().start()
    
    def step(self, action: int) -> Tuple[float, float, bool, Dict[str, object]]:
        self.update_agent(action)
        self.update_display()
        return super().step(action)

    def update_display(self) -> None:
        if self.display:
            self.is_quit()
            self.surface.fill(BLUE)
            self.draw_track()
            pygame.display.update()
            self.game_clock.tick(self.fps)

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
        else:
            raise Exception(f"Non-human operator not yet implemented")
        
        if action == FORWARD:
            self.agent_position += self.agent_speed
            if self.agent_position >= self.agent_position_threshold:
                self.agent_position = 0

            self.track_position_y_index += self.track_position_y_increment
            # if we reach the curve's end we invert it's incrementation to exit it
            if self.track_position_y_index >= self.track_view_length:
                self.track_position_y_index = self.track_view_length
                self.track_position_y_increment *= -1
            # if we totally exit the curve we invert it's incrementation to enter it again
            # we also invert the curve's direction to change the way
            elif self.track_position_y_index < -1:
                self.track_position_y_increment *= -1
                self.track_curve_direction *= -1
            a = self.get_last_track_position()
            s = self.scale_track_value(a)
            x = self.track_position_x - (s * self.track_curve_direction)
            _logger.info(f"track_list:last,scaled,x_pos> {a}, {s}, {x}")
            self.track_position_x -= self.scale_track_value(a) * self.track_curve_direction
        elif action == LEFT:
            if self.track_position_x - self.track_position_x_increment >= -self.track_border:
                self.track_position_x -= self.track_position_x_increment
        elif action == RIGHT:
            if self.track_position_x + self.track_position_x_increment <= self.track_border:
                self.track_position_x += self.track_position_x_increment

    def draw_track(self) -> None:
        z = 0
        dz = 0
        track_list = []
        road_position = self.agent_position
        road_position_threshold = self.agent_position_threshold//2
        for i in range(self.plane_center_y, 0, -1):
            if road_position < road_position_threshold:
                strip = self.strip_light
                road = self.road_light
            else:
                strip = self.strip_dark
                road = self.road_dark

            track_y = i + self.plane_center_y
            track_x = self.track[self.track_position_y_index - i] * self.track_curve_direction if self.track_position_y_index >= i else 0
            track_list.append((track_x, track_y))
            # self.surface.blit(strip, (0, track_y)) 
            self.surface.blit(road, (track_x, track_y), (self.track_position_x, i, self.width, 1))

            dz += self.road_increment_z
            z += dz
            road_position += self.road_position_speed + z
            if road_position >= self.agent_position_threshold:
               road_position = 0

        # for i in range(self.track_view_length):
        #     pos = self.track[i]
        #     pygame.draw.circle(self.surface, BLACK, (pos, self.plane_center_y), 3)
        t2 = self.scale_track_value(track_list[-1][0]) * self.track_curve_direction
        pygame.draw.circle(self.surface, RED, (self.track[-1], self.plane_center_y), 5)
        pygame.draw.circle(self.surface, RED, (self.width - self.track[-1], self.plane_center_y), 5)
        pygame.draw.circle(self.surface, WHITE, (t2, self.plane_center_y), 5)
        pygame.draw.circle(self.surface, WHITE, (self.width - t2, self.plane_center_y), 5)

    def get_last_track_position(self) -> float:
        track_list = []
        for i in range(self.plane_center_y, 0, -1):
            track_x = self.track[self.track_position_y_index - i] * self.track_curve_direction if self.track_position_y_index >= i else 0
            track_y = self.plane_center_y + i
            track_list.append((track_x, track_y))
        return track_list[len(track_list) - 1][0]

    def scale_track_value(self, value) -> float:
        return (((value - self.track_value_min) * (self.track_value_max_scaled - self.track_value_min_scaled)) / (self.track_value_max - self.track_value_min)) + self.track_value_min_scaled

    def log_init_debug(self) -> None:
        for i in range(self.track_view_length):
            _logger.info(f"track value: {i}, {self.track[i]}")
        # --------------------------------------------------
        pass

    def log_debug(self) -> None:
        # track_list = list(range(self.plane_center_y, 0, -1))
        # _logger.info(f"track first,last: ({track_list[0]}, {track_list[len(track_list) - 1]})")
        # _logger.info("-" * 50)
        # --------------------------------------------------
        # a = self.track_position_y_index
        # b = self.track_curve_direction

        # first = self.plane_center_y - 1
        # first_idx = a - first
        # last = 0
        # last_idx = a - last

        # # _logger.info(f"first_idx,last_idx, {first_idx}, {last_idx}")
        # first_x = self.track[first_idx] * b if a >= first else 0
        # last_x = self.track[last_idx] * b if a >= last else 0
        # _logger.info(f"track first,last: ({first_x}, {last_x})")
        # --------------------------------------------------
        pass
