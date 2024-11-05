from typing import List
from PIL import Image

from customTypes.Pokemon import Pokemon

class FeatureReduction(object):
    def __init__(self):
        pass

    def downSample(self, data: List[Pokemon], newWidth: int, newHeight: int) -> List[Pokemon]:
        '''
            Returns a new list of Pokemon, where each Pokemon's image will be resized to (newWidth, newHeight).
        '''
        
        newData: List[Pokemon] = []

        for pokemon in data:
            copy: Pokemon = pokemon.getCopy()
            copy.setImage(copy.image.resize((newWidth, newHeight)))
            newData.append(copy)

        return newData