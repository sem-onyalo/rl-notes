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

GRASS = "./assets/grass.png"
ROAD = "./assets/road.png"
AGENT = "./assets/agent_car.png"

_logger = logging.getLogger(RACE_TRACK_MDP)

class RaceTrackMDP(PyGameMDP):
    """
    Race track MDP based off of https://raytomely.itch.io/p.
    """
    def __init__(self) -> None:
        super().__init__()

        self.fps = 60
        self.width = 640
        self.height = 480

        self.n_state = self.width * self.height
        self.n_action = 1 # FORWARD
        self.d_state = (self.width, self.height)

        self.view_angle = 90
        self.move_speed = 35
        self.move_x = int(self.move_speed * math.cos(math.radians(self.view_angle)))
        self.move_y = -int(self.move_speed * math.sin(math.radians(self.view_angle)))
        self.agent_position = [self.width//4, 224]

        fov = 60
        wall_height = 64
        self.ray_angle = 90
        self.resolution = 1
        self.agent_height = wall_height/2
        self.plane_center_x = self.width//2
        self.plane_center_y = self.height//2
        self.to_plane_dist = int((self.width/2) / math.tan(math.radians(fov/2)))

        # self.init_display()

    def init_display(self) -> None:
        super().init_display()
        grass_image = pygame.image.load(GRASS).convert()
        self.grass_image = pygame.transform.scale(grass_image, (self.width, grass_image.get_height()))
        self.road_image = pygame.image.load(ROAD).convert()
        self.road_width = self.road_image.get_width()
        self.road_height = self.road_image.get_height()
        agent = pygame.image.load(AGENT).convert_alpha()
        self.agent = pygame.transform.scale(agent, (self.width//4, 82))
        self.ground = pygame.Surface((self.width, self.height//2)).convert()
        self.ground.fill((0, 100, 0))

    def start(self) -> np.ndarray:
        super().start()
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
            self.raycast()
            pygame.display.update()
            self.game_clock.tick(self.fps)

    def update_agent(self, action:int) -> None:
        if self.operator == HUMAN:
            action = -1
            if self.check_input():
                pressed = pygame.key.get_pressed()
                if pressed[K_UP]:
                    action = FORWARD
        else:
            raise Exception(f"Non-human operator not yet implemented")
        
        if action == FORWARD:
            self.agent_position[0] += self.move_x
            self.agent_position[1] += self.move_y
            if self.agent_position[1] < 0:
                self.agent_position[1] = 5000

    def raycast(self) -> None:
        cos_beta = math.cos(math.radians(self.view_angle - self.ray_angle))
        # cos_angle=math.cos(math.radians(self.ray_angle))
        sin_angle=-math.sin(math.radians(self.ray_angle))
        wall_bottom = self.height
        while wall_bottom > self.plane_center_y + 10:
            wall_bottom -= self.resolution
            # row at floor point-row of center
            row = wall_bottom - self.plane_center_y
            #straight distance from player to the intersection with the floor 
            straight_p_dist = (self.agent_height / row * self.to_plane_dist)
            #true distance from player to floor
            to_floor_dist=(straight_p_dist / cos_beta)
            # coordinates (x,y) of the floor
            # ray_x = int(player_pos[0] + (to_floor_dist * cos_angle))
            ray_y = int(self.agent_position[1] + (to_floor_dist * sin_angle))
            # floor_x = (ray_x % road_width)
            floor_y = (ray_y % self.road_height)
            # the road and grass positions
            slice_width = int(self.road_width / to_floor_dist * self.to_plane_dist)
            slice_x = (self.plane_center_x) - (slice_width//2)
            row_slice = self.road_image.subsurface(0, floor_y, self.road_width, 1)
            row_slice = pygame.transform.scale(row_slice, (slice_width, self.resolution))
            self.surface.blit(self.grass_image, (0, wall_bottom), (0, floor_y, self.width, self.resolution))
            self.surface.blit(row_slice,(slice_x, wall_bottom))
            self.surface.blit(self.agent, (240, 320))
