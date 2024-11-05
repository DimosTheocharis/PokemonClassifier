from typing import List, Tuple
import time

from algorithms.KNN import KNNalgorithm
from algorithms.NCC import NCCalgorithm

from components.DataLoader import DataLoader
from components.DataSplitter import DataSplitter
from components.FeatureReduction import FeatureReduction
from components.DataAugmentation import DataAugmentation

from customTypes.Pokemon import Pokemon
from customTypes.Settings import FeatureReductionSettings

class Scenarios(object):
    filePathForData: str = "data/pokemonData.csv"
    filePathForImages: str = "data/pokemonImages"
    '''
        This class sets up the environment for running specific cases about the project. For example one
        scenario could be running KNN with the data and one other scenario to run KNN with reduced data 
        (feature reduction) or with augmented data (data augmentation)
    '''
    @staticmethod
    def KNNScenario(
        k: int, 
        featureReductionSettings: FeatureReductionSettings | None = None, 
        augmentData: bool = False
    ):
        '''
            Runs KNN algorithm with the following settings: \n
            * {k} neighbors
        '''
        dataLoader: DataLoader = DataLoader()
        dataSplitter: DataSplitter = DataSplitter()
        featureReduction: FeatureReduction = FeatureReduction()
        dataAugmentation: DataAugmentation = DataAugmentation()
        
        data: List[Pokemon] = dataLoader.readPokemonData(Scenarios.filePathForData, Scenarios.filePathForImages)

        if (featureReductionSettings):
            data = featureReduction.downSample(data, featureReductionSettings['newWidth'], featureReductionSettings['newHeight'])
        
        splitted = dataSplitter.trainTestSplit(data)

        trainSet: List[Pokemon] = splitted[0]
        testSet: List[Pokemon] = splitted[1]

        if (augmentData):
            trainSet = dataAugmentation.augment(trainSet)
            testSet = dataAugmentation.augment(testSet)

        knn: KNNalgorithm = KNNalgorithm(k, trainSet)
        knnSuccessRate: float = knn.run(testSet)

        print(f"""
              Algorithm = KNN,
              k={k},
              Image dimensions = {data[0].image.size},
              Train-set size = {len(trainSet)},
              Test-set size = {len(testSet)},
              Data augmentation = {augmentData},
              Success rate = {round(knnSuccessRate, 2) * 100}%""")




    def NCCScenario(
            featureReductionSettings: FeatureReductionSettings | None = None,
            augmentData: bool = False
        ):
        ''' 
            Runs NCC algorithm with the following settings: \n
        '''
        dataLoader: DataLoader = DataLoader()
        dataSplitter: DataSplitter = DataSplitter()
        featureReduction: FeatureReduction = FeatureReduction()
        dataAugmentation: DataAugmentation = DataAugmentation()

        data: List[Pokemon] = dataLoader.readPokemonData(Scenarios.filePathForData, Scenarios.filePathForImages)

        if (featureReductionSettings):
            data = featureReduction.downSample(data, featureReductionSettings['newWidth'], featureReductionSettings['newHeight'])

        splitted = dataSplitter.trainTestSplit(data)

        trainSet: List[Pokemon] = splitted[0]
        testSet: List[Pokemon] = splitted[1]

        if (augmentData):
            trainSet = dataAugmentation.augment(trainSet)
            testSet = dataAugmentation.augment(testSet)

        ncc = NCCalgorithm(trainSet)
        nccSuccessRate = ncc.run(testSet)

        print(f"""Algorithm = NCC, \n 
                  Image dimensions = {data[0].image.size}, 
                  Train-set size = {len(trainSet)},  
                  Test-set size = {len(testSet)},
                  Data augmentation = {augmentData},
                  Success rate = {round(nccSuccessRate, 2) * 100}%""")

        


