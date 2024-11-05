
import random
from typing import List, Dict, Tuple

from customTypes.Pokemon import Pokemon
from customTypes.PokemonType import PokemonType

class DataSplitter(object):
    def __init__(self):
        self.pokemonsPerType: Dict[PokemonType, int] = {}
        self.trainSetSize: float = 0.8

        self.__initializePokemonsPerType()


    ######################################## PUBLIC METHODS ########################################
    def trainTestSplit(self, data: List[Pokemon]) -> Tuple[List[Pokemon], List[Pokemon]]:
        '''
            Randomly seperates the dataset into 2 sets: \n
            * Train set (default 80%)
            * Test set (default 20%)
        '''
        self.__calculateTotalPokemonsPerType(data)

        trainSet: List[Pokemon] = []
        testSet: List[Pokemon] = []

        for pokemonType in PokemonType:
            totalPokemonsForTrain: int = round(self.pokemonsPerType[pokemonType] * self.trainSetSize)

            # Find the pokemons that have as primary type the current {pokemonType}
            allPokemons: List[Pokemon] = [pokemon for pokemon in data if pokemon.type1 == pokemonType]

            # Randomly reorder the pokemons that have the current {pokemonType}
            random.shuffle(allPokemons)

            # The first {totalPokemonsForTrain} pokemons belong to the train set
            trainSet.extend(allPokemons[0: totalPokemonsForTrain])
            testSet.extend(allPokemons[totalPokemonsForTrain:])

        if (self.__validateTrainAndTestSets(trainSet, testSet) == False): 
            print("!!!!!!! The splitting isn't proper! There are duplicate pokemons.")
            return ([], [])


        return trainSet, testSet
    

    def reset(self) -> None:
        self.pokemonsPerType = {}
        self.__initializePokemonsPerType()


    ######################################## PRIVATE METHODS ########################################
    def __calculateTotalPokemonsPerType(self, data: List[Pokemon]) -> None: 
        '''
            Counts the number of pokemons that have as primary type each pokemon type
        '''
        for pokemon in data:
            self.pokemonsPerType[pokemon.type1] += 1

    def __initializePokemonsPerType(self) -> None:
        '''
            Initializes the counter for each pokemon type to zero.
        '''
        for pokemonType in PokemonType:
            self.pokemonsPerType[pokemonType] = 0


    def __validateTrainAndTestSets(self, trainSet: List[Pokemon], testSet: List[Pokemon]) -> bool:
        '''
            Counts the number of pokemons that have as primary type each pokemon type
        '''
        # A dictionary that keeps track of how many times each pokemon is found in trainSet and testSet together.
        counters: Dict[str, int] = {}
        
        for pokemon in trainSet:
            if (pokemon.name in counters):
                counters[pokemon.name] += 1
            else:
                counters[pokemon.name] = 1

        for pokemon in testSet:
            if (pokemon.name in counters):
                counters[pokemon.name] += 1
            else:
                counters[pokemon.name] = 1

        for name, encountings in counters.items():
            if encountings > 1:
                return False

        return True
