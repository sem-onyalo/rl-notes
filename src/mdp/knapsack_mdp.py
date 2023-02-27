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
    target:float
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

        self.starts = 0
        self.max_value = 0
        
    def init_display(self) -> None:
        super().init_display()
        if self.display:
            self.font_header = pygame.font.Font(pygame.font.get_default_font(), 18)
            self.font_title = pygame.font.Font(pygame.font.get_default_font(), 26)

    def reset(self) -> None:
        self.starts += 1
        self.current_weight = 0
        self.current_value = 0
        self.selected_item = 0
        self.selected_items:List[Item] = []
        self.step_result:StepResult = None

    def start(self) -> float:
        self.reset()
        self.update_display()
        return super().start()

    def step(self, action: int) -> Tuple[float, float, bool, Dict[str, object]]:
        self.update_state(action)
        self.update_display()
        return None, None, self.step_result.is_terminal, { "action": action }
    
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
        
        reward = 0
        is_terminal = False
        if action == NEXT:
            self.selected_item += 1
            if self.selected_item >= self.n_action:
                self.selected_item = 0
        elif action == PREV:
            self.selected_item -= 1
            if self.selected_item < 0:
                self.selected_item = self.n_action - 1
        elif action != None:
            if self.current_weight + self.data.items[action].weight <= self.data.capacity:
                self.current_weight += self.data.items[action].weight
                self.current_value += self.data.items[action].value
                self.selected_items.append(self.data.items[action])
                self.max_value = max([self.max_value, self.current_value])
                is_terminal = self.current_weight == self.data.capacity
            else:
                is_terminal = True

        self.step_result = StepResult()
        self.step_result.is_terminal = is_terminal
        self.step_result.reward = reward

    def draw_assets(self) -> None:
        gauge_cell_height = 150
        knapsack_cell_width = 540
        padding_vert = 50
        padding_horz = 20
        rect_width = 5
        text_margin_x = 10
        header_margin_y = 30
        text_margin_y = 20

        text_s = "RL Agent Maximizing Value"
        text_x = self.width//2
        text_y = padding_vert
        text_v = self.font_title.render(text_s, True, BLUE_LIGHT)
        text_r = text_v.get_rect()
        text_r.center = (text_x, text_y)
        self.surface.blit(text_v, text_r)
        title_text_h = text_v.get_height()

        text_s = f"{self.max_value:.1f}"
        text_x = padding_horz
        text_y = padding_vert + title_text_h + header_margin_y
        text_v = self.font_header.render(text_s, True, BLUE_LIGHT)
        text_r = text_v.get_rect()
        text_r.topleft = (text_x, text_y)
        self.surface.blit(text_v, text_r)

        text_s = f"Episode: {self.starts}"
        text_x = self.width - padding_horz
        text_y = padding_vert + title_text_h + header_margin_y
        text_v = self.font_header.render(text_s, True, BLUE_LIGHT)
        text_r = text_v.get_rect()
        text_r.topright = (text_x, text_y)
        self.surface.blit(text_v, text_r)
        header_text_w = text_v.get_width()
        header_text_h = text_v.get_height()

        if self.step_result != None and self.step_result.is_terminal:
            text_s = "TERMINAL"
            text_x = self.width - padding_horz - header_text_w - text_margin_x
            text_y = padding_vert + title_text_h + header_margin_y
            text_v = self.font_header.render(text_s, True, BLUE_MEDIUM)
            text_r = text_v.get_rect()
            text_r.topright = (text_x, text_y)
            self.surface.blit(text_v, text_r)

        padding_top = padding_vert + title_text_h + header_margin_y + header_text_h + header_margin_y
        gauge_h = self.draw_gauge("Capacity", padding_horz, padding_top, text_margin_y, self.current_weight / self.data.capacity, f"{self.current_weight/self.data.capacity:.0%}")

        padding_top = padding_vert + title_text_h + header_margin_y + header_text_h + header_margin_y + gauge_h + text_margin_y
        value_pct = self.current_value / self.max_value if self.max_value > 0 else self.current_value
        gauge_h = self.draw_gauge("Value", padding_horz, padding_top, text_margin_y, value_pct, f"{self.current_value:.1f}")

        knapsack_p1 = (padding_horz, gauge_cell_height + padding_vert * 3)
        knapsack_p2 = (padding_horz, self.height - padding_vert)
        knapsack_p3 = (knapsack_cell_width - padding_horz, self.height - padding_vert)
        knapsack_p4 = (knapsack_cell_width - padding_horz, gauge_cell_height + padding_vert * 3)
        pygame.draw.line(self.surface, DARK, knapsack_p1, knapsack_p2, width=rect_width)
        pygame.draw.line(self.surface, DARK, knapsack_p2, knapsack_p3, width=rect_width)
        pygame.draw.line(self.surface, DARK, knapsack_p3, knapsack_p4, width=rect_width)

        selected_padding = 10
        selected_margin = 10
        selected_w = knapsack_p3[0] - knapsack_p1[0]
        selected_h = 50
        selected_x = knapsack_cell_width + padding_horz
        selected_y = self.height - padding_vert - selected_h - ((selected_h) * self.selected_item)
        pygame.draw.rect(self.surface, BLUE_LIGHT, (selected_x, selected_y, selected_w, selected_h), width=rect_width)

        item_w = knapsack_p3[0] - knapsack_p1[0] - selected_padding * 2
        item_h = selected_h - selected_padding * 2

        item_x = padding_horz + rect_width + selected_padding//2
        self.draw_items(self.selected_items, item_w, item_h, item_x, rect_width, selected_padding, selected_margin, padding_vert)

        item_x = knapsack_cell_width + padding_horz + rect_width + selected_padding//2
        self.draw_items(self.data.items, item_w, item_h, item_x, rect_width, selected_padding, selected_margin, padding_vert)

    def draw_items(self, items:List[Item], item_w, item_h, item_x, rect_width, selected_padding, selected_margin, vert_padding) -> None:
        for i, item in enumerate(items):
            item_y = self.height - vert_padding - ((item_h + rect_width + selected_padding//2) * (i + 1)) - (selected_margin * i)
            pygame.draw.rect(self.surface, BLUE_MEDIUM, (item_x, item_y, item_w, item_h), width=0)
            text_s = f"Weight: {item.weight:.1f}, Value: {item.value:.1f}"
            text_x = item_x + item_w//2
            text_y = item_y + item_h//2
            text_v = self.font_values.render(text_s, True, DARK)
            text_r = text_v.get_rect()
            text_r.center = (text_x, text_y)
            self.surface.blit(text_v, text_r)

    def draw_gauge(self, title, padding_left, padding_top, text_margin_y, value, value_str) -> float:
        text_s = title
        text_x = padding_left
        text_y = padding_top
        text_v = self.font_values.render(text_s, True, BLUE_LIGHT)
        text_r = text_v.get_rect()
        text_r.topleft = (text_x, text_y)
        self.surface.blit(text_v, text_r)
        title_h = text_v.get_height()

        text_s = value_str
        text_x = self.width - padding_left
        text_y = padding_top
        text_v = self.font_values.render(text_s, True, BLUE_LIGHT)
        text_r = text_v.get_rect()
        text_r.topright = (text_x, text_y)
        self.surface.blit(text_v, text_r)

        gauge_x = padding_left
        gauge_y = padding_top + title_h + text_margin_y
        gauge_w = self.width - padding_left * 2
        gauge_h = 2
        pygame.draw.rect(self.surface, DARK, (gauge_x, gauge_y, gauge_w, gauge_h))

        gauge_x = padding_left
        gauge_y = padding_top + title_h + text_margin_y
        gauge_w = (self.width - padding_left * 2) * value
        gauge_h = 2
        pygame.draw.rect(self.surface, BLUE_LIGHT, (gauge_x, gauge_y, gauge_w, gauge_h))

        return title_h + text_margin_y + gauge_h

    def init_state(self, data:KnapsackData) -> None:
        if data == None:
            data = self.get_knapsack_data_test()
            self.n_state = data.capacity + 1
            self.n_action = len(data.items)
            self.d_state = (self.n_action, 1)

        self.data = data

    def get_knapsack_data_test(self) -> KnapsackData:
        items = []
        items.append(Item(weight=12., value=4.,  name="item-1"))
        items.append(Item(weight=1.,  value=2.,  name="item-2"))
        items.append(Item(weight=2.,  value=2.,  name="item-3"))
        items.append(Item(weight=1.,  value=1.,  name="item-4"))
        items.append(Item(weight=4.,  value=10., name="item-5"))
        capacity = 15
        target = 32
        data = KnapsackData()
        data.capacity = capacity
        data.target = target
        data.items = items
        return data
