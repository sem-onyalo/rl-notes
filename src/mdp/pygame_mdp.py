import sys
import time

import pygame
from pygame.locals import QUIT

from .mdp import MDP
from constants import HUMAN

TEXT_COLOUR = (36, 113, 163)

class PyGameMDP(MDP):
    width:int
    height:int
    display:bool
    operator:str
    surface:pygame.Surface
    game_clock:pygame.time.Clock
    font_values:pygame.font.Font

    def __init__(self) -> None:
        super().__init__()

        self.debounce_val = 100
        self.debounce = time.time_ns()

    def start(self) -> float:
        assert self.operator != None, "Set agent operator parameter before starting"

    def init_display(self) -> None:
        if self.display:
            pygame.init()
            self.game_clock = pygame.time.Clock()
            self.surface = pygame.display.set_mode((self.width, self.height))
            self.font_values = pygame.font.Font(pygame.font.get_default_font(), 14)
            pygame.display.set_caption("PyGame MDP")

    def is_quit(self) -> None:
        for event in pygame.event.get():
            if event.type == QUIT:
                self.quit()

    def quit(self) -> None:
        pygame.quit()
        sys.exit()

    def set_operator(self, operator: str) -> None:
        super().set_operator(operator)
        if self.operator == HUMAN:
            self.display = True
            self.init_display()

    def check_input(self) -> bool:
        if ((time.time_ns() - self.debounce) / 1e6) < self.debounce_val:
            return False
        else:
            self.debounce = time.time_ns()
            return True

    def set_policy(self, policy) -> None:
        # TODO: remove this, the MDP should not have access to the policy object
        #       this is only used to show values in the grid target MDP
        #       need to find another way to show values in the grid target MDP
        pass
