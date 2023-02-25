class Item:
    def __init__(self, weight:int, value:int, name:str) -> None:
        self.weight = weight
        self.value = value
        self.name = name

    def __str__(self) -> str:
        return f"'{self.name}'(W{self.weight},V{self.value})"

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, self.__class__) 
            and self.weight == __o.weight 
            and self.value == __o.value 
            and self.name == __o.name
        )