import logging
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pygame
from pygame.locals import *

from .pygame_mdp import PyGameMDP
from .pygame_mdp import TEXT_COLOUR
from constants import *
from model import Item
from model import StepResult

BLUE_DARK   = (  0,  52,  89)
BLUE_LIGHT  = (  0, 168, 232)
BLUE_MEDIUM = (  0, 126, 167)
DARK        = (  0,  23,  31)

NEXT = -1
PREV = -2

_logger = logging.getLogger(KNAPSACK_MDP)

class KnapsackData:
    capacity:float
    items:List[Item]

class KnapsackMDP(PyGameMDP):
    """
    An MDP representing a knapsack problem (https://en.wikipedia.org/wiki/Knapsack_problem).
    """
    def __init__(self, data:KnapsackData=None) -> None:
        super().__init__()

        self.init_state(data)
        assert isinstance(self.data, KnapsackData), f"Error: must set data attribute in constructor"

        self.fps = 60
        self.width = 1080
        self.height = 1080

        self.display = True
        self.selected_item = 0
        self.selected_items:List[Item] = []

    def start(self) -> float:
        self.selected_items.clear()
        self.update_display()
        return super().start()
    
    def step(self, action: int) -> Tuple[float, float, bool, Dict[str, object]]:
        self.update_state(action)
        self.update_display()
        return super().step(action)
    
    def update_display(self) -> None:
        if self.display:
            self.is_quit()
            self.surface.fill(BLUE_DARK)
            self.draw_assets()
            pygame.display.update()
            self.game_clock.tick(self.fps)

    def update_state(self, action:int) -> None:
        if self.operator == HUMAN:
            action = None
            if self.check_input():
                pressed = pygame.key.get_pressed()
                if pressed[K_UP]:
                    action = NEXT
                elif pressed[K_DOWN]:
                    action = PREV
                elif pressed[K_RETURN]:
                    action = self.selected_item
        else:
            raise Exception(f"Non-human operator not yet implemented")
        
        if action == NEXT:
            self.selected_item += 1
            if self.selected_item >= self.n_action:
                self.selected_item = 0
        elif action == PREV:
            self.selected_item -= 1
            if self.selected_item < 0:
                self.selected_item = self.n_action - 1
        elif action != None:
            self.selected_items.append(self.data.items[action])

    def draw_assets(self) -> None:
        grid_1_width = 200
        grid_2_width = 440
        grid_3_width = 440
        vert_padding = 200
        horz_padding = 20
        rect_width = 5

        # pygame.draw.line(self.surface, DARK, (grid_1_width, 0), (grid_1_width, self.height), width=5)
        # pygame.draw.line(self.surface, DARK, (grid_1_width + grid_2_width, 0), (grid_1_width + grid_2_width, self.height), width=rect_width)

        gauge_x = horz_padding
        guage_y = vert_padding
        guage_w = grid_1_width - horz_padding * 2
        guage_h = self.height - vert_padding * 2
        pygame.draw.rect(self.surface, DARK, (gauge_x, guage_y, guage_w, guage_h), width=rect_width)

        knapsack_p1 = (grid_1_width + horz_padding, vert_padding)
        knapsack_p2 = (grid_1_width + horz_padding, self.height - vert_padding)
        knapsack_p3 = (grid_1_width + grid_2_width - horz_padding, self.height - vert_padding)
        knapsack_p4 = (grid_1_width + grid_2_width - horz_padding, vert_padding)
        pygame.draw.line(self.surface, DARK, knapsack_p1, knapsack_p2, width=rect_width)
        pygame.draw.line(self.surface, DARK, knapsack_p2, knapsack_p3, width=rect_width)
        pygame.draw.line(self.surface, DARK, knapsack_p3, knapsack_p4, width=rect_width)

        selected_padding = 10
        selected_margin = 10
        selected_w = knapsack_p3[0] - knapsack_p1[0]
        selected_h = 50
        selected_x = grid_1_width + grid_2_width + horz_padding
        selected_y = self.height - vert_padding - selected_h - ((selected_h) * self.selected_item)
        pygame.draw.rect(self.surface, BLUE_LIGHT, (selected_x, selected_y, selected_w, selected_h), width=rect_width)

        item_w = knapsack_p3[0] - knapsack_p1[0] - selected_padding * 2
        item_h = selected_h - selected_padding * 2

        item_x = grid_1_width + horz_padding + rect_width + selected_padding//2
        self.draw_items(self.selected_items, item_w, item_h, item_x, rect_width, selected_padding, selected_margin, vert_padding)

        item_x = grid_1_width + grid_2_width + horz_padding + rect_width + selected_padding//2
        self.draw_items(self.data.items, item_w, item_h, item_x, rect_width, selected_padding, selected_margin, vert_padding)

    def draw_items(self, items:List[Item], item_w, item_h, item_x, rect_width, selected_padding, selected_margin, vert_padding) -> None:
        for i, item in enumerate(items):
            item_y = self.height - vert_padding - ((item_h + rect_width + selected_padding//2) * (i + 1)) - (selected_margin * i)
            pygame.draw.rect(self.surface, BLUE_MEDIUM, (item_x, item_y, item_w, item_h), width=0)
            text_s = f"Weight: {item.weight}, Value: {item.value}"
            text_x = item_x + item_w//2
            text_y = item_y + item_h//2
            text_v = self.font_values.render(text_s, True, DARK)
            text_r = text_v.get_rect()
            text_r.center = (text_x, text_y)
            self.surface.blit(text_v, text_r)

    def init_state(self, data:KnapsackData) -> None:
        if data == None:
            data = self.get_knapsack_data_test()
            self.n_state = data.capacity + 1
            self.n_action = len(data.items)
            self.d_state = (self.n_action, 1)

        self.data = data

    def get_knapsack_data_test(self) -> KnapsackData:
        items = []
        items.append(Item(weight=12, value=4.,  name="item-1"))
        items.append(Item(weight=1,  value=2.,  name="item-2"))
        items.append(Item(weight=2,  value=2.,  name="item-3"))
        items.append(Item(weight=1,  value=1.,  name="item-4"))
        items.append(Item(weight=4,  value=10., name="item-5"))
        capacity = 15
        data = KnapsackData()
        data.capacity = capacity
        data.items = items
        return data
