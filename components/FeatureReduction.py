from typing import List
from PIL import Image

from customTypes.Pokemon import Pokemon

class FeatureReduction(object):
    def __init__(self):
        pass

    def downSample(self, data: List[Pokemon], factor: int) -> List[Pokemon]:
        '''
            Returns a new list of Pokemon, where each Pokemon's image will be {factor} times sampled down.
            For example, having an image of 120x120 pixels and sampling down with a factor of 4 times,
            will get you a 30x30 pixels image.
        '''
        
        newData: List[Pokemon] = []

        for pokemon in data:
            copy: Pokemon = pokemon.getCopy()
            copy.setImage(copy.image.reduce(factor))

            newData.append(copy)

        return newData