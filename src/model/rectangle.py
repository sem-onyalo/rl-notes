from .point import Point

class Rectangle:
    def __init__(self, topleft:Point, topright:Point, bottomright:Point, bottomleft:Point) -> None:
        self.topleft = topleft
        self.topright = topright
        self.bottomright = bottomright
        self.bottomleft = bottomleft
