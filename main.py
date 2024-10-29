import numpy as np
from PIL import Image

from components.DataLoader import DataLoader
from components.DataSplitter import DataSplitter
from algorithms.KNN import KNNalgorithm

from customTypes.Pokemon import Pokemon
from customTypes.PokemonType import PokemonType

dataLoader = DataLoader()

data = dataLoader.readPokemonData("data/pokemonData.csv")

dataSplitter = DataSplitter()

trainSet, testSet = dataSplitter.trainTestSplit(data)
 
img = Image.fromarray(trainSet[0].image, "RGB")
img.show()

newImg = img.reduce(4)

newImg.show()


#knn = KNNalgorithm(100, trainSet)

# print(knn.vote(
#     [
#         (Pokemon("Dimos", PokemonType.Fire), 5700),
#         (Pokemon("Vasilis", PokemonType.Steel), 5500),
#         (Pokemon("Gotsis", PokemonType.Fire, PokemonType.Ground), 6000)
#     ],
# ))


#successRate = knn.run(testSet)

#print(f"The success rate of the test set with K-NN is {successRate}")