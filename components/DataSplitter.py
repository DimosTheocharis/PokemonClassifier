
import random
from typing import List, Dict, Tuple

from customTypes.Pokemon import Pokemon
from customTypes.PokemonType import PokemonType

class DataSplitter(object):
    '''
        This class is responsible for providing methods that split the entire datasets into smaller sub-sets
    '''
    def __init__(self):
        self.__pokemonsPerType: Dict[PokemonType, int] = {}

        self.__initializePokemonsPerType()


    ######################################## PUBLIC METHODS ########################################
    def trainTestValidationSplit(
            self, 
            data: List[Pokemon],
            trainSetSize: float = 0.8,
            testSetSize: float = 0.1,
            validationSetSize: float = 0.1
        ) -> Tuple[List[Pokemon], List[Pokemon], List[Pokemon]]:
        '''
            Randomly seperates the dataset into 3 sets: \n
            * Train set (default 80%)
            * Test set (default 10%) 
            * Validation set (default 10%) \n
            Then it randomly shuffles the pokemons in each set in order for each pokemon type 
            to be present in the whole set
        '''
        self.__calculateTotalPokemonsPerType(data)

        trainSet: List[Pokemon] = []
        testSet: List[Pokemon] = []
        validationSet: List[Pokemon] = []

        for pokemonType in PokemonType:
            # Find the number of pokemons that should consist the train set
            totalPokemonsForTrain: int = round(self.__pokemonsPerType[pokemonType] * trainSetSize)

            # Find the number of pokemons that should consist the test set
            totalPokemonsForTest: int = round(self.__pokemonsPerType[pokemonType] * testSetSize)

            # Find the pokemons that have as primary type the current {pokemonType}
            allPokemons: List[Pokemon] = [pokemon for pokemon in data if pokemon.type1 == pokemonType]

            # Randomly reorder the pokemons that have the current {pokemonType}
            random.shuffle(allPokemons)

            # The first {totalPokemonsForTrain} pokemons belong to the train set
            trainSet.extend(allPokemons[0: totalPokemonsForTrain])

            # The next {totalPokemonsForTest} pokemons belong to the test set
            testSet.extend(allPokemons[totalPokemonsForTrain : totalPokemonsForTrain + totalPokemonsForTest])

            # The rest pokemons belong to the validation set
            validationSet.extend(allPokemons[totalPokemonsForTrain + totalPokemonsForTest:])

        if (self.__validateTrainTestValidationSets(trainSet, testSet, validationSet) == False): 
            print("!!!!!!! The splitting isn't proper! There are duplicate pokemons.")
            return ([], [])
        
        # Randomly shuffle the pokemon in the trainSet & testSet
        random.shuffle(trainSet)
        random.shuffle(testSet)
        random.shuffle(validationSet)

        return trainSet, testSet, validationSet
    
    ######################################## PRIVATE METHODS ########################################
    def __calculateTotalPokemonsPerType(self, data: List[Pokemon]) -> None: 
        '''
            Counts the number of pokemons that have as primary type each pokemon type
        '''
        for pokemon in data:
            self.__pokemonsPerType[pokemon.type1] += 1

    def __initializePokemonsPerType(self) -> None:
        '''
            Initializes the counter for each pokemon type to zero.
        '''
        for pokemonType in PokemonType:
            self.__pokemonsPerType[pokemonType] = 0


    def __validateTrainTestValidationSets(self, trainSet: List[Pokemon], testSet: List[Pokemon], validationSet: List[Pokemon]) -> bool:
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

        for pokemon in validationSet:
            if (pokemon.name in counters):
                counters[pokemon.name] += 1
            else:
                counters[pokemon.name] = 1

        for name, encountings in counters.items():
            if encountings > 1:
                return False

        return True
