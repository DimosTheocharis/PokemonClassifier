from typing import List
import numpy as np
from numpy.typing import NDArray
from numpy import uint8
from PIL import Image

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
    

    def getImagePixels(self) -> NDArray[uint8]:
        return np.array(self.image).flatten()
    
    def setImage(self, image: Image) -> None:
        self.image = image 

    def getCopy(self) -> "Pokemon": 
        copiedPokemon = Pokemon(self.name, self.type1, self.type2)
        copiedImage: Image = Image.fromarray(self.getImagePixels().reshape(self.image.size[0], self.image.size[0], 3), "RGB")
        copiedPokemon.setImage(copiedImage)

        return copiedPokemon