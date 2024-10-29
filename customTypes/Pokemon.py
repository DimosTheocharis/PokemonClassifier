from numpy.typing import NDArray

from customTypes.PokemonType import PokemonType

class Pokemon(object):
    def __init__(self, name: str, type1: PokemonType, type2: PokemonType=None):
        self.name: str = name
        self.type1: PokemonType = type1
        self.type2: PokemonType = type2


    def __str__(self):
        return f"{{ name: {self.name}, type1: {self.type1}, type2: {self.type2} }}"

    def __repr__(self):
        return f"Pokemon(name={self.name}, type1={self.type1}, type2={self.type2})"
    
    def setImage(self, image: NDArray) -> None:
        self.image = image 