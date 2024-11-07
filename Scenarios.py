from typing import List, Tuple, Dict
import time

from algorithms.KNN import KNNalgorithm
from algorithms.NCC import NCCalgorithm
from algorithms.MLP import MultiLayerPerceptron

from components.DataLoader import DataLoader
from components.DataSplitter import DataSplitter
from components.FeatureReduction import FeatureReduction
from components.DataAugmentation import DataAugmentation
from components.DataPreparation import DataPreparation

from customTypes.Pokemon import Pokemon
from customTypes.Settings import FeatureReductionSettings
from customTypes.NeuralNetworkTypes import LayerType, ActivationFunctionType

class Scenarios(object):
    filePathForData: str = "data/pokemonData.csv"
    filePathForImages: str = "data/pokemonImages"
    filePathForTrainSet: str = "data/trainSet.csv"
    filePathForTestSet: str = "data/testSet.csv"
    '''
        This class sets up the environment for running specific cases about the project. For example one
        scenario could be running KNN with the data and one other scenario to run KNN with reduced data 
        (feature reduction) or with augmented data (data augmentation)
    '''
    @staticmethod
    def KNNScenario(
        k: int, 
        featureReductionSettings: FeatureReductionSettings | None = None, 
        augmentData: bool = False,
        readSavedData: bool = True
    ):
        '''
            Runs KNN algorithm with the following settings: \n
            * {k} neighbors
        '''
        # Begin timer
        startTime = time.time()

        featureReduction: FeatureReduction = FeatureReduction()
        dataAugmentation: DataAugmentation = DataAugmentation()
        
        trainSet, testSet = Scenarios.__readAndSplitData(readSavedData)

        print([pokemon.name for pokemon in trainSet[-5:]])

        if (augmentData):
            trainSet = dataAugmentation.augment(trainSet)
            testSet = dataAugmentation.augment(testSet)

        print([pokemon.name for pokemon in trainSet[-5:]])

        if (featureReductionSettings):
            trainSet = featureReduction.downSample(trainSet, featureReductionSettings['newWidth'], featureReductionSettings['newHeight'])
            testSet = featureReduction.downSample(testSet, featureReductionSettings['newWidth'], featureReductionSettings['newHeight'])

        print([pokemon.name for pokemon in trainSet[-5:]])
        dimensionality: int = trainSet[0].image.size

        knn: KNNalgorithm = KNNalgorithm(k, trainSet)
        knnSuccessRate: float = knn.run(testSet)

        # Ternimate timer
        endTime = time.time()

        print(f"""
              Algorithm = KNN,
              k={k},
              Image dimensions = {dimensionality},
              Train-set size = {len(trainSet)},
              Test-set size = {len(testSet)},
              Data augmentation = {augmentData},
              Success rate = {round(knnSuccessRate * 100, 2)}%
              Total time = {round(endTime - startTime)} seconds""")

    @staticmethod
    def NCCScenario(
            featureReductionSettings: FeatureReductionSettings | None = None,
            augmentData: bool = False,
            readSavedData: bool = True
        ):
        ''' 
            Runs NCC algorithm with the following settings: \n
        '''
        # Begin timer
        startTime = time.time()

        featureReduction: FeatureReduction = FeatureReduction()
        dataAugmentation: DataAugmentation = DataAugmentation()

        trainSet, testSet = Scenarios.__readAndSplitData(readSavedData)
        
        print([pokemon.name for pokemon in trainSet[-5:]])

        if (augmentData):
            trainSet = dataAugmentation.augment(trainSet)
            testSet = dataAugmentation.augment(testSet)

        print([pokemon.name for pokemon in trainSet[-5:]])

        if (featureReductionSettings):
            trainSet = featureReduction.downSample(trainSet, featureReductionSettings['newWidth'], featureReductionSettings['newHeight'])
            testSet = featureReduction.downSample(testSet, featureReductionSettings['newWidth'], featureReductionSettings['newHeight'])

        print([pokemon.name for pokemon in trainSet[-5:]])

        dimensionality: int = trainSet[0].image.size

        ncc = NCCalgorithm(trainSet)
        nccSuccessRate = ncc.run(testSet)

        # Ternimate timer
        endTime = time.time()

        print(f"""Algorithm = NCC, \n 
                  Image dimensions = {dimensionality}, 
                  Train-set size = {len(trainSet)},  
                  Test-set size = {len(testSet)},
                  Data augmentation = {augmentData},
                  Success rate = {round(nccSuccessRate * 100, 2)}%
                  Total time = {round(endTime - startTime)} seconds""")
        

    @staticmethod
    def MLPScenario(
            featureReductionSettings: FeatureReductionSettings | None = None,
            augmentData: bool = False,
            readSavedData: bool = True
        ):
        featureReduction: FeatureReduction = FeatureReduction()
        dataAugmentation: DataAugmentation = DataAugmentation()
        dataPreperation: DataPreparation = DataPreparation()

        trainSet, testSet = Scenarios.readAndSplitData(readSavedData)

        if (augmentData):
            trainSet = dataAugmentation.augment(trainSet)
            testSet = dataAugmentation.augment(testSet)


        if (featureReductionSettings):
            trainSet = featureReduction.downSample(trainSet, featureReductionSettings['newWidth'], featureReductionSettings['newHeight'])
            testSet = featureReduction.downSample(testSet, featureReductionSettings['newWidth'], featureReductionSettings['newHeight'])


        print(f"First pokemon in trainSet: {trainSet[0]}")
        print(f"Last pokemon in trainSet: {trainSet[-1]}")
        print(f"First pokemon in testSet: {testSet[0]}")
        print(f"Last pokemon in testSet: {testSet[-1]}")

        print(len(trainSet))

        network: MultiLayerPerceptron = MultiLayerPerceptron(learningRate=0.001, layers=[
                (LayerType.Linear, 512, ActivationFunctionType.ReLu),
                (LayerType.Linear, 18, ActivationFunctionType.Sigmoid),
            ],)
        xTrain, yTrain = dataPreperation.prepare(trainSet)
        xTest, yTest = dataPreperation.prepare(testSet)

        network.train(xTrain, yTrain)
        network.test(xTest, yTest)
        

    def countPokemonTypesScenario():
        dataLoader: DataLoader = DataLoader()
        
        data: List[Pokemon] = dataLoader.readPokemonData(Scenarios.filePathForData, Scenarios.filePathForImages)

        counter: Dict[str, int] = {}

        for pokemon in data:
            if pokemon.type1.name in counter:
                counter[pokemon.type1.name] += 1
            else:
                counter[pokemon.type1.name] = 1

            if pokemon.type2 == None:
                continue

            if pokemon.type2.name in counter:
                counter[pokemon.type2.name] += 1
            else:
                counter[pokemon.type2.name] = 1

        return counter

    @staticmethod
    def readAndSplitData(readSavedData: bool) -> Tuple[List[Pokemon], List[Pokemon]]:
        dataLoader: DataLoader = DataLoader()
        dataSplitter: DataSplitter = DataSplitter()

        if (readSavedData):
            trainSet: List[Pokemon] = dataLoader.readPokemonData(Scenarios.filePathForTrainSet, Scenarios.filePathForImages)
            testSet: List[Pokemon] = dataLoader.readPokemonData(Scenarios.filePathForTestSet, Scenarios.filePathForImages)
        else:   
            data: List[Pokemon] = dataLoader.readPokemonData(Scenarios.filePathForData, Scenarios.filePathForImages)
            splitted = dataSplitter.trainTestSplit(data)

            trainSet: List[Pokemon] = splitted[0]
            testSet: List[Pokemon] = splitted[1]

        return trainSet, testSet