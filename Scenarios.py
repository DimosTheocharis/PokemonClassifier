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
from components.CrossValidation import CrossValidation

from customTypes.Pokemon import Pokemon
from customTypes.Settings import FeatureReductionSettings
from customTypes.NeuralNetworkTypes import LayerType, ActivationFunctionType

class Scenarios(object):
    filePathForData: str = "data/pokemonData.csv"
    filePathForImages: str = "data/pokemonImages"
    filePathForTrainSet: str = "data/trainSet.csv"
    filePathForTestSet: str = "data/testSet.csv"
    filePathForValidationSet: str = "data/validationSet.csv"
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
        
        trainSet, testSet, _ = Scenarios.readAndSplitData(readSavedData)

        if (featureReductionSettings):
            trainSet = featureReduction.downSample(trainSet, featureReductionSettings['newWidth'], featureReductionSettings['newHeight'])
            testSet = featureReduction.downSample(testSet, featureReductionSettings['newWidth'], featureReductionSettings['newHeight'])

        if (augmentData):
            trainSet = dataAugmentation.augment(trainSet)
            testSet = dataAugmentation.augment(testSet)
        
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

        trainSet, testSet, _ = Scenarios.readAndSplitData(readSavedData)

        if (featureReductionSettings):
            trainSet = featureReduction.downSample(trainSet, featureReductionSettings['newWidth'], featureReductionSettings['newHeight'])
            testSet = featureReduction.downSample(testSet, featureReductionSettings['newWidth'], featureReductionSettings['newHeight'])
        

        if (augmentData):
            trainSet = dataAugmentation.augment(trainSet)
            testSet = dataAugmentation.augment(testSet)

        dimensionality: int = trainSet[0].image.size

        ncc = NCCalgorithm(trainSet)
        nccSuccessRate = ncc.run(testSet)

        # Ternimate timer
        endTime = time.time()

        print(f"""
              Algorithm = NCC, 
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

        trainSet, testSet, validationSet = Scenarios.readAndSplitData(readSavedData, generateValidationSet=True)

        if (featureReductionSettings):
            trainSet = featureReduction.downSample(trainSet, featureReductionSettings['newWidth'], featureReductionSettings['newHeight'])
            testSet = featureReduction.downSample(testSet, featureReductionSettings['newWidth'], featureReductionSettings['newHeight'])
            validationSet = featureReduction.downSample(validationSet, featureReductionSettings['newWidth'], featureReductionSettings['newHeight'])

        if (augmentData):
            trainSet = dataAugmentation.augment(trainSet)
            testSet = dataAugmentation.augment(testSet)
            validationSet = dataAugmentation.augment(validationSet)

        xTrain, yTrain = dataPreperation.prepare(trainSet)
        xTest, yTest = dataPreperation.prepare(testSet)
        xValidation, yValidation = dataPreperation.prepare(validationSet)

        dimensionality: int = xTrain.size(dim=1)

        network: MultiLayerPerceptron = MultiLayerPerceptron(learningRate=0.01, layers=[
                (LayerType.Linear, 256, ActivationFunctionType.ReLu),
                (LayerType.Linear, 18, ActivationFunctionType.Sigmoid),
        ], dimensionality=dimensionality)


        print(network)

        network.train(xTrain, yTrain, True, xValidation, yValidation)
        network.test(xTest, yTest)


    def crossValidationScenario(
        featureReductionSettings: FeatureReductionSettings | None = None,
        augmentData: bool = False,
        readSavedData: bool = True
    ):
        featureReduction: FeatureReduction = FeatureReduction()
        dataAugmentation: DataAugmentation = DataAugmentation()
        dataPreperation: DataPreparation = DataPreparation()

        trainSet, testSet, _ = Scenarios.readAndSplitData(readSavedData, generateValidationSet=False)
        
        if (featureReductionSettings):
            trainSet = featureReduction.downSample(trainSet, featureReductionSettings['newWidth'], featureReductionSettings['newHeight'])
            testSet = featureReduction.downSample(testSet, featureReductionSettings['newWidth'], featureReductionSettings['newHeight'])

        if (augmentData):
            trainSet = dataAugmentation.augment(trainSet)
            testSet = dataAugmentation.augment(testSet)

        xTrain, yTrain = dataPreperation.prepare(trainSet)
        xTest, yTest = dataPreperation.prepare(testSet)

        crossValidation: CrossValidation = CrossValidation(10, xTrain, yTrain)
        
        architectureValues: List[List[Tuple[LayerType, int, ActivationFunctionType]]] = [
            # [
            #     (LayerType.Linear, 512, ActivationFunctionType.ReLu),
            #     (LayerType.Linear, 256, ActivationFunctionType.ReLu),
            #     (LayerType.Linear, 18, ActivationFunctionType.Sigmoid)
            # ],
            # [
            #     (LayerType.Linear, 1025, ActivationFunctionType.ReLu),
            #     (LayerType.Linear, 512, ActivationFunctionType.ReLu),
            #     (LayerType.Linear, 256, ActivationFunctionType.ReLu),
            #     (LayerType.Linear, 18, ActivationFunctionType.Sigmoid)
            # ]
            [
                (LayerType.Linear, 1024, ActivationFunctionType.ReLu),
                (LayerType.Linear, 18, ActivationFunctionType.Sigmoid)
            ],
            [
                (LayerType.Linear, 512, ActivationFunctionType.ReLu),
                (LayerType.Linear, 18, ActivationFunctionType.Sigmoid)
            ],
            [
                (LayerType.Linear, 256, ActivationFunctionType.ReLu),
                (LayerType.Linear, 18, ActivationFunctionType.Sigmoid)
            ],
            [
                (LayerType.Linear, 64, ActivationFunctionType.ReLu),
                (LayerType.Linear, 18, ActivationFunctionType.Sigmoid)
            ]
        ]

        epochValues: List[int] = [50]
        learningRateValues: List[float] = [0.001, 0.0001, 0.005]

        crossValidation.run(architectureValues, epochValues, learningRateValues)

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
    def readAndSplitData(readSavedData: bool, generateValidationSet: bool) -> Tuple[List[Pokemon], List[Pokemon], List[Pokemon]]:
        '''
            Reads and splits the pokemon data in sub-sets as following:
            * {readSavedData} = True, {generateValidationSet} = True -> Reads train (80%), test (10%) and validation (10%) data from files
            * {readSavedData} = True, {generateValidationSet} = False -> Reads train (80%), test (20%) and validation (0%) data from files 
            by combining testFile and validationFile into test data
            * {readSavedData}
            
        '''
        dataLoader: DataLoader = DataLoader()
        dataSplitter: DataSplitter = DataSplitter()

        if (readSavedData):
            if (generateValidationSet):    
                trainSet: List[Pokemon] = dataLoader.readPokemonData(Scenarios.filePathForTrainSet, Scenarios.filePathForImages)
                testSet: List[Pokemon] = dataLoader.readPokemonData(Scenarios.filePathForTestSet, Scenarios.filePathForImages)
                validationSet: List[Pokemon] = dataLoader.readPokemonData(Scenarios.filePathForValidationSet, Scenarios.filePathForImages)
            else:
                trainSet: List[Pokemon] = dataLoader.readPokemonData(Scenarios.filePathForTrainSet, Scenarios.filePathForImages)
                testSet: List[Pokemon] = dataLoader.readPokemonData(Scenarios.filePathForTestSet, Scenarios.filePathForImages)
                validationSet: List[Pokemon] = dataLoader.readPokemonData(Scenarios.filePathForValidationSet, Scenarios.filePathForImages)

                testSet.extend(validationSet)
                validationSet = []
        else:   
            data: List[Pokemon] = dataLoader.readPokemonData(Scenarios.filePathForData, Scenarios.filePathForImages)
            if (generateValidationSet):
                splitted = dataSplitter.trainTestValidationSplit(data)
            else:
                splitted = dataSplitter.trainTestValidationSplit(data, trainSetSize=0.8, testSetSize=0.2)

            trainSet: List[Pokemon] = splitted[0]
            testSet: List[Pokemon] = splitted[1]
            validationSet: List[Pokemon] = splitted[2]

        return trainSet, testSet, validationSet