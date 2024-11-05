from typing import List, Dict
from numpy.typing import NDArray
import numpy as np
from math import sqrt
from PIL import Image

from customTypes.Pokemon import Pokemon
from customTypes.PokemonType import PokemonType
from algorithms.KNN import KNNalgorithm

class NCCalgorithm(object):
    ''' 
        This class implements the Nearest-Class-Centroid algorithm
    '''
    def __init__(self, trainSet: List[Pokemon]):
        self.__classCentroids = self.__findClassCentroids(trainSet)
        self.__knn = KNNalgorithm(1, self.__classCentroids)


    def run(self, testSet: List[Pokemon]) -> int:
        ''' 
            Runs the Nearest-Class-Centroid algorithm for the given testSet. The method will return the classification 
            success rate (from 0.0 to 1.0)
        '''
        return self.__knn.run(testSet)

    ######################################## PRIVATE METHODS ########################################
    def __findClassCentroids(self, trainSet: List[Pokemon]) -> List[Pokemon]:
        '''
            Calculates the centroid pokemon of each class (pokemon type) based on the image. 
            The centroid pokemon has as image, the average color of the images of the pokemons that belong 
            to that class, for every pixel. 
        '''
        classCentroids: List[Pokemon] = []

        # Seperate the train set into classes
        pokemonClasses: Dict[PokemonType, List[PokemonType]] = self.__seperatePokemonsBasedOnType(trainSet)

        for pokemonType, members in pokemonClasses.items():
            # Compute the centroid of each class
            classCentroid = self.__computeClassCentroid(members, pokemonType)
            classCentroids.append(classCentroid)
        
        return classCentroids

    def __seperatePokemonsBasedOnType(self, pokemons: List[Pokemon]) -> Dict[PokemonType, List[PokemonType]]:
        '''
            Creates a dictionary of key-value pairs, where the key is each Pokemon type and the value is a list
            with the members of the class (pokemon-type) ie, of all pokemons that have this particular type as type1
        '''
        # Initialize the members of each class to empty list
        pokemonClasses: Dict[PokemonType, List[PokemonType]] = {pokemonType: [] for pokemonType in PokemonType}

        for pokemon in pokemons:
            pokemonClasses[pokemon.type1].append(pokemon)


        return pokemonClasses
    

    def __computeClassCentroid(self, members: List[Pokemon], pokemonType: PokemonType) -> Pokemon:
        '''
            Generates a new Pokemon that acts as the centroid pokemon of the given class. The centroid pokemon
            has the same {pokemonType} with the members of the class it represents, and as image the 
            average image of the {members}
            The members should have the same first pokemon type! (type1)
        '''
        centroid: Pokemon = Pokemon(pokemonType.name + "-centroid", pokemonType, None)

        avgImage: NDArray[np.float16] = members[0].getImagePixels().astype(np.float16) / len(members)

        for x in range(1, len(members)):
            avgImage += members[x].getImagePixels().astype(np.float16) / len(members)

        imageSize = round(sqrt(len(avgImage) / 3))
        
        centroid.setImage(Image.fromarray(avgImage.reshape(imageSize, imageSize, 3).astype(np.uint8), "RGB"))
        
        return centroid
