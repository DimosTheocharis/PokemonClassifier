from typing import List, Tuple
import torch
import numpy as np

from customTypes.Pokemon import Pokemon
from customTypes.PokemonType import PokemonType

class DataPreparation(object):
    def __init__(self):
        pass

    def prepare(self, dataSet: List[Pokemon]) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
            Generates two tensors inside a tuple from the given pokemon list. The first tensor contains
            the X data that can be feeded to a neural network, and the second tensor contains the y data,
            or labels (targets). More specifically: if n = dataset size: \n
            X -> torch.Tensor(n, totalPixels) for every pixel of every pokemon's image \n
            y -> torch.Tensor(n, ) for the class \n 

            where totalPixels = Pokemon.image.width * Pokemon.image.height * 3 (for RGB channels)

        '''
        X: List[torch.Tensor] = []
        y: List[torch.Tensor] = []

        for pokemon in dataSet:
            row: torch.Tensor = torch.tensor(pokemon.getImagePixels() / 255, dtype=torch.float32) # Normalize the data. From pixels (0 - 255) to (0, 1)
            row = torch.round(row, decimals=2) # Round to 2 decimals
            X.append(row)

            targetVector: torch.Tensor = self.__createTargetVector(pokemon)
            y.append(targetVector)

        # Transform the variables X, y from a list of 1-dimensional tensors, to a 2-dimensional tensor
        X: torch.Tensor = torch.stack(X, dim=0)
        y: torch.Tensor = torch.stack(y, dim=0)

        return (X, y)
    

    def find2MostLikelyPokemonTypes(self, probabilities: torch.Tensor) -> Tuple[PokemonType, PokemonType]:
        '''
            Given a tensor of {probabilities} for each Pokemon type, this method returns the 2 types with
            the highest probability, in a tuple.
        '''
        probabilities: List[float] = probabilities.tolist()
        maxProbIndex1: int = 0
        maxProbIndex2: int = 1

        # Make sure that maxProb1 contains the 1st highest probability and the maxProb2 contains the 2nd one.
        if (probabilities[maxProbIndex1] < probabilities[maxProbIndex2]):
            test = maxProbIndex1
            maxProbIndex1 = maxProbIndex2
            maxProbIndex2 = test

        for i in range(2, len(probabilities)):
            if (probabilities[i] > probabilities[maxProbIndex1]):
                maxProbIndex2 = maxProbIndex1
                maxProbIndex1 = i
            elif (probabilities[i] > probabilities[maxProbIndex2]):
                maxProbIndex2 = i


        return (PokemonType(maxProbIndex1), PokemonType(maxProbIndex2))



    def __createTargetVector(self, pokemon: Pokemon) -> torch.Tensor:
        '''
            Creates a torch.Tensor which contains as many binary digits (0 or 1) as the total classes (pokemon types) \n
            The number 1 in the j index means that the Pokemon belongs to the j-th pokemon type \n
            The number 0  means that the pokemon does not belong to the corresponding pokemon type \n

            For example if pokemon types are [Fire, Grass, Water, Electric] then the vector [0, 1, 0, 1] means that the
            given {pokemon} is Grass-type and Electric-type
        '''
        
        targets: List[int] = [0 for pokemonType in PokemonType]
        
        targets[pokemon.type1.value] = 1

        # Check if the given {pokemon} has second type
        if (pokemon.type2 != None):
            targets[pokemon.type2.value] = 1


        return torch.tensor(targets, dtype=torch.float32)


    