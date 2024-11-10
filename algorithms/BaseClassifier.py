from typing import List
import numpy as np

from customTypes.Pokemon import Pokemon


class BaseClassifier():
    def __init__(self, trainSet: List[Pokemon]):
        self._trainSet = trainSet

    ######################################## PROTECTED METHODS ########################################
    def _calculateDistance(self, pokemonA: Pokemon, pokemonB: Pokemon) -> int:
        '''
            Computes the "distance" metric between the given pokemons: {pokemonA}, {pokemonB}.
            The distance between 2 pokemons is the l2 norm of the matrix that results as the difference
            of the pixels of these images in every point
        '''        
        return round(np.linalg.norm(pokemonA.getImagePixels().astype(np.int16) - pokemonB.getImagePixels().astype(np.int16)), 2)