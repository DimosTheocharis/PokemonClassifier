from typing import List

from customTypes.Pokemon import Pokemon

class FeatureReduction(object):
    '''
        This class is responsible for providing methods that reduct a sample's size. Currently, the only method
        for feature reduction is sampling down the image of a Pokemon (sample).
    '''
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