from typing import List, Tuple, Dict
import torch
import time

from algorithms.KNN import KNNalgorithm
from algorithms.NCC import NCCalgorithm
from algorithms.MLP import MultiLayerPerceptron
from algorithms.BaseClassifier import BaseClassifier

from components.DataLoader import DataLoader
from components.DataSplitter import DataSplitter
from components.FeatureReduction import FeatureReduction
from components.DataAugmentation import DataAugmentation
from components.DataPreparation import DataPreparation
from components.CrossValidation import CrossValidation
from components.Logger import Logger

from customTypes.Pokemon import Pokemon
from customTypes.PokemonType import PokemonType
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
        
        trainSet, testSet, _ = Scenarios.readAndSplitData(readSavedData, generateValidationSet=False)

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

        logger: Logger = Logger("logs/KNN results", appendTimestamp=False)
        messages: List[str] = [
            f"k={k}",
            f"\nImage dimensions = {dimensionality}",
            f"\nTrain-set size = {len(trainSet)}",
            f"\nTest-set size = {len(testSet)}",
            f"\nData augmentation = {augmentData}",
            f"\nRead saved data = {readSavedData}",
            f"\nSuccess rate = {round(knnSuccessRate * 100, 2)}%",
            f"\nTotal time = {round(endTime - startTime)} seconds"
        ]

        logger.logData(messages, True)

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

        trainSet, testSet, _ = Scenarios.readAndSplitData(readSavedData, generateValidationSet=False)

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

        logger: Logger = Logger("logs/NCC results", appendTimestamp=False)
        messages: List[str] = [
            "Algorithm = NCC",
            f"\nImage dimensions = {dimensionality}",
            f"\nTrain-set size = {len(trainSet)}",
            f"\nTest-set size = {len(testSet)}",
            f"\nData augmentation = {augmentData}",
            f"\nSuccess rate = {round(nccSuccessRate * 100, 2)}%",
            f"\nTotal time = {round(endTime - startTime)} seconds"
        ]

        logger.logData(messages, printToConsole=True)
        

    @staticmethod
    def MLPScenario(
            featureReductionSettings: FeatureReductionSettings | None = None,
            augmentData: bool = False,
            readSavedData: bool = True
        ):
        featureReduction: FeatureReduction = FeatureReduction()
        dataAugmentation: DataAugmentation = DataAugmentation()
        dataPreperation: DataPreparation = DataPreparation()
        logger: Logger = Logger("logs/MLP", appendTimestamp=True)

        trainSet, testSet, validationSet = Scenarios.readAndSplitData(readSavedData, generateValidationSet=False)

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

        dimensionality: int = xTrain.size(dim=1)

        network: MultiLayerPerceptron = MultiLayerPerceptron(learningRate=0.005, layers=[
                (LayerType.Linear, 1024, ActivationFunctionType.ReLu),
                (LayerType.Linear, 18, ActivationFunctionType.Sigmoid),
        ], dimensionality=dimensionality, epochs=30)

        network.train(xTrain, yTrain, False)
        loss = network.test(xTest, yTest)

        messages: List[str] = [
             "\t \t \t \t MLP session data after I changed crop functions to include more space to the center",
             "\n" + str(network),
             f"\n* Read saved data: {readSavedData}",
             f"\n* Augment data: {augmentData}",
             f"\n* Dimensionality of data: {dimensionality}",
             f"\n* Train-set size: {xTrain.size(dim=0)}",
             f"\n* Test-set size: {xTest.size(dim=0)}",
             f"\nLoss: {round(loss, 2)}",
        ]

        logger.logData(messages)

        return network

    @staticmethod
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
            [
                (LayerType.Linear, 512, ActivationFunctionType.ReLu),
                (LayerType.Linear, 256, ActivationFunctionType.ReLu),
                (LayerType.Linear, 18, ActivationFunctionType.Sigmoid)
            ],
            [
                (LayerType.Linear, 512, ActivationFunctionType.Tanh),
                (LayerType.Linear, 256, ActivationFunctionType.Tanh),
                (LayerType.Linear, 18, ActivationFunctionType.Sigmoid)
            ],
            # [
            #     (LayerType.Linear, 1025, ActivationFunctionType.ReLu),
            #     (LayerType.Linear, 512, ActivationFunctionType.ReLu),
            #     (LayerType.Linear, 256, ActivationFunctionType.ReLu),
            #     (LayerType.Linear, 128, ActivationFunctionType.ReLu),
            #     (LayerType.Linear, 18, ActivationFunctionType.Sigmoid)
            # ],
            # [
            #     (LayerType.Linear, 1024, ActivationFunctionType.Tanh),
            #     (LayerType.Linear, 18, ActivationFunctionType.Sigmoid)
            # ],
            # [
            #     (LayerType.Linear, 1024, ActivationFunctionType.ReLu),
            #     (LayerType.Linear, 18, ActivationFunctionType.Sigmoid)
            # ],
            # [
            #     (LayerType.Linear, 512, ActivationFunctionType.ReLu),
            #     (LayerType.Linear, 18, ActivationFunctionType.Sigmoid)
            # ],
            # [
            #     (LayerType.Linear, 256, ActivationFunctionType.ReLu),
            #     (LayerType.Linear, 18, ActivationFunctionType.Sigmoid)
            # ],
            # [
            #     (LayerType.Linear, 64, ActivationFunctionType.ReLu),
            #     (LayerType.Linear, 18, ActivationFunctionType.Sigmoid)
            # ]
        ]

        epochValues: List[int] = [40]
        learningRateValues: List[float] = [0.001, 0.0001, 0.005]
        learningRateValues = [0.001, 0.005, 0.0001]

        crossValidation.run(architectureValues, epochValues, learningRateValues)

    @staticmethod
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
    def checkPokemonDistanceScenario(pokemonA: str, pokemonB: str) -> None:
        '''
            Calculates the distance between {pokemonA} and {pokemonB}. The arguments are strings and 
            represent the names of the pokemons, so at first, it loads these Pokemons from the database.
        '''
        dataLoader: DataLoader = DataLoader()
        
        data: List[Pokemon] = dataLoader.readPokemonData(Scenarios.filePathForData, Scenarios.filePathForImages)
        
        baseClassifier: BaseClassifier = BaseClassifier(data)

        pokemonA: Pokemon = [pokemon for pokemon in data if pokemon.name == pokemonA][0]
        pokemonB: Pokemon = [pokemon for pokemon in data if pokemon.name == pokemonB][0]

        distance: float = baseClassifier._calculateDistance(pokemonA, pokemonB)

        print(f"Distance({pokemonA.name}, {pokemonB.name}) = {distance}")

    @staticmethod
    def testNeuralNetworkModel(model: MultiLayerPerceptron, pokemonType: PokemonType, width: int, height: int):
        '''
            Runs the neural network model for all Pokemons that belong to the given {pokemonType}, that is,
            they have either type1 = {pokemonType} or type2 = {pokemonType}
            It returns the accuracy of the model. The model classifies 100% correctly a Pokemon, if the 2 classes
            with the highest probabilities are the same with the sample's type1 and type2 (if it indeed has second type).
            If the model succeeds in predicting only one of the 2 classes, then the accuracy for the particular
            sample is 50%. \n

            Note: The found Pokemons will be affected by feature reduction so as their images are in the given
            dimensions {width} x {height}. This is because the given {model} is trained with data that have
            specific dimensionality.
        '''
        dataPreparation: DataPreparation = DataPreparation()
        featureReduction: FeatureReduction = FeatureReduction()
        cluster: List[Pokemon] = Scenarios.findPokemonCluster(pokemonType)

        cluster = featureReduction.downSample(cluster, width, height)

        counter: float = 0
        
        for pokemon in cluster:
            # Transform the information about the Pokemon into pytorch tensors that my {model} understands
            xTest, yTest = dataPreparation.prepare([pokemon])

            # Run model and get the probabilities of the current {pokemon} belonging to each class
            probabilities: torch.Tensor = model.testRaw(xTest)

            # Get the best predictions
            typeA, typeB = dataPreparation.find2MostLikelyPokemonTypes(probabilities[0])

            if (pokemon.type2 == None):
                if (pokemon.type1 == typeA or pokemon.type1 == typeB):
                    counter += 1
            else:
                if (pokemon.type1 == typeA or pokemon.type1 == typeB):
                    counter += 0.5

                if (pokemon.type2 == typeA or pokemon.type2 == typeB):
                    counter += 0.5

        accuracy: float = round(counter / len(cluster) * 100, 2)
        print(f"Accuracy: {accuracy}%")

        pass

    @staticmethod
    def findPokemonCluster(pokemonType: PokemonType) -> List[Pokemon]:
        '''
            Returns all Pokemon that have the given {pokemonType} either as type1 or as type2.
            All these Pokemon belong to the same cluster.
        '''
        dataLoader: DataLoader = DataLoader()
        data: List[Pokemon] = dataLoader.readPokemonData(Scenarios.filePathForData, Scenarios.filePathForImages)

        filteredPokemons: List[Pokemon] = [pokemon for pokemon in data if pokemon.type1 == pokemonType or 
                                           (pokemon.type2 != None and pokemon.type2 == pokemonType)]

        return filteredPokemons

    @staticmethod
    def getPokemonAugmentation(pokemonName: str) -> List[Pokemon]:
        dataAugmentation: DataAugmentation = DataAugmentation()

        pokemon: Pokemon = Scenarios.getPokemon(pokemonName)

        if (pokemon):
            return dataAugmentation.augment([pokemon])
        

    @staticmethod
    def getPokemon(pokemonName: str) -> Pokemon:
        dataLoader: DataLoader = DataLoader()

        data = dataLoader.readPokemonData(Scenarios.filePathForData, Scenarios.filePathForImages)

        pokemon: Pokemon = [pokemon for pokemon in data if pokemon.name == pokemonName][0]

        return pokemon
        

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