import numpy as np
from PIL import Image
from typing import List
import random

from components.DataLoader import DataLoader
from components.DataSplitter import DataSplitter
from components.FeatureReduction import FeatureReduction

from algorithms.KNN import KNNalgorithm

from customTypes.Pokemon import Pokemon
from customTypes.PokemonType import PokemonType

dataLoader = DataLoader()
featureReduction: FeatureReduction = FeatureReduction()
dataSplitter = DataSplitter()

data: List[Pokemon] = dataLoader.readPokemonData("data/pokemonData.csv")
trainSet, testSet = dataSplitter.trainTestSplit(data)

knn = KNNalgorithm(3, trainSet)

successRate = knn.run(testSet)

print(f"The success rate of the test set with K-NN, k={knn.k} and image dimensions = {data[0].image.size} is {successRate}")



# DOWN SAMPLING TO 30X30

newTrainSet = featureReduction.downSample(trainSet, 4)
newTestSet = featureReduction.downSample(testSet, 4)

dataSplitter.reset()

knn = KNNalgorithm(3, newTrainSet)

successRate = knn.run(newTestSet)


print(f"The success rate of the test set with K-NN, k={knn.k} and image dimensions = {newTestSet[0].image.size} is {successRate}")