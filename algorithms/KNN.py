from typing import List, Tuple, Dict

from algorithms.BaseClassifier import BaseClassifier
from customTypes.Pokemon import Pokemon
from customTypes.PokemonType import PokemonType


class KNNalgorithm(BaseClassifier):
    ''' 
        This class implements the K-Nearest-Neighbor algorithm
    '''
    def __init__(self, k: int, trainSet: List[Pokemon]):
        super().__init__(trainSet)
        self.__k = k


    def run(self, testSet: List[Pokemon]) -> float:
        ''' 
            Runs the K-Nearest-Neighbor for the given testSet. The method will return the classification 
            success rate (from 0.0 to 1.0). A sample (pokemon) from the test set is correctly classified 
            if the most common class (pokemonType) of the k nearest neighbors is the same as the sample's 
            type1 or type2
        '''

        correctClassifications: int = 0

        for index, testPokemon in enumerate(testSet):
            kNearestNeighbors: List[Tuple[Pokemon, int]] = self.__findKNearestNeighbors(testPokemon)
            votedPokemonType: PokemonType = self.__vote(kNearestNeighbors)

            if (votedPokemonType == testPokemon.type1 or votedPokemonType == testPokemon.type2):
                correctClassifications += 1
        
        return correctClassifications / len(testSet) if len(testSet) > 0 else 0


    ######################################## PRIVATE METHODS ########################################
    def __findKNearestNeighbors(self, testPokemon: Pokemon) -> List[Tuple[Pokemon, int]]:
        '''
            Finds the k Pokemons from the train set that are nearest to the given {testPokemon} 
        '''
        kNearestNeighbors: List[Tuple[Pokemon, int]] = []

        # Initialize the k nearest neighbors to the first k pokemons in the train set
        for i in range(self.__k):
            distance = self._calculateDistance(testPokemon, self._trainSet[i])
            kNearestNeighbors.append((self._trainSet[i], distance))
        
        for i in range(self.__k, len(self._trainSet)):
            trainPokemon: Pokemon = self._trainSet[i]
            distance: int = self._calculateDistance(testPokemon, trainPokemon)

            indexOfTheMostDistantNeighbor: int = self.__findIndexOfTheMostDistantNeighbor(kNearestNeighbors)

            if (distance < kNearestNeighbors[indexOfTheMostDistantNeighbor][1]):
                kNearestNeighbors[indexOfTheMostDistantNeighbor] = (trainPokemon, distance)


        return kNearestNeighbors
    

    def __findIndexOfTheMostDistantNeighbor(self, kNearestNeighbors: List[Tuple[Pokemon, int]]) -> int:
        '''
            Given the k nearest neighbors, this method finds the farthest neighbor and returns his index.
        '''
        index = 0

        for i in range(1, len(kNearestNeighbors)):
            if (kNearestNeighbors[i][1] > kNearestNeighbors[index][1]):
                index = i

        return index
    

    def __vote(self, kNearestNeighbors: List[Tuple[Pokemon, int]]) -> PokemonType:
        '''
            Will predict the pokemonType as following:
            For every pokemonType, the method will compute a value. The pokemonType that will be assigned the biggest value
            will be returned. The value will be computed by adding up normalized distance complement factors. For example:
            neighbor1: Fire, 5700(distance)
            neighbor2: Steel, 5500
            neighbor3: Fire & Ground, 6000

            Total distance = 17200 \n
            Fire -> (17200 - 5700) / 17200 + (17200 - 6000) / 17200 = 1.32 \n
            Steel -> (17200 - 5500) / 17200 = 0.68 \n
            Ground -> (17200 - 6000) / 17200 = 0.65 \n

            So the pokemonType = Fire will be returned
        '''
        votes: Dict[PokemonType, float] = {}
        
        # Initialize the vote for each pokemon type
        for type in PokemonType:
            votes[type] = 0


        # Calculate the sum of the distance of each neighbor
        totalDistance: int = 0
        for neighbor in kNearestNeighbors:
            totalDistance += neighbor[1]

        # Implement the voting
        for neighbor in kNearestNeighbors:
            votes[neighbor[0].type1] += round((totalDistance - neighbor[1]) / totalDistance, 2)
            
            # If neighbor has 2nd type
            if (neighbor[0].type2):
                votes[neighbor[0].type2] += round((totalDistance - neighbor[1]) / totalDistance, 2)

        votedPokemonType: PokemonType = None
        # Find the pokemon type with the biggest vote

        for type, vote in votes.items():
            if (not votedPokemonType):
                votedPokemonType = type
            else:
                if (vote > votes[votedPokemonType]):
                    votedPokemonType = type

        return votedPokemonType
    