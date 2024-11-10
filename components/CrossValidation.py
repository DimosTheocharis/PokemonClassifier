import torch
from typing import List, Tuple
import time

from algorithms.MLP import MultiLayerPerceptron
from components.Logger import Logger
from customTypes.NeuralNetworkTypes import LayerType, ActivationFunctionType

class CrossValidation(object):
    '''
        This class is responsible for testing multiple MLP neural network architectures using the cross
        validation technique. The goal is to find the most optimal model for the given dataset. \n
        k-Cross validation concerns the splitting of the dataset to k equal-size subsets (folds).
        Each of these neural networks will be trained and tested k times. Each time they will be trained
        in k-1 subsets and tested in the subset that was not used in training. Each subset will act exactly
        one time as a test-set. The architecture that will get the minimum average loss, will be selected
        as the best model for the given dataset {X}, {y}
    '''
    def __init__(self, k: int, X: torch.Tensor, y: torch.Tensor) -> None:
        self.__k: int = k
        self.__dimensionality = X.size(dim=1)
        self.__multitude = X.size(dim=0) # How many data consist the dataset (k - 1 train sets and 1 validation set)
        self.__X: torch.Tensor = X
        self.__y: torch.Tensor = y

    def run(
            self,
            architectureValues: List[List[Tuple[LayerType, int, ActivationFunctionType]]],
            epochValues: List[int],
            learningRateValues: List[float]
        ) -> None:
        '''
            This methods creates an MLP neural network for every possible combination of the given parameters:
            * {architectureValues} -> Each architecture provide information about its layers in the format
            (type of layer, number or neurons, type of activation function)
            * {epochValues} -> Possible values about how many iterations the model will run in the training mode.
            * {learningRateValues} -> Possible values about how fast/slow the weights follow the 
            slope of the error-derivative 
        '''
        logger: Logger = Logger("logs/Run_5", appendTimestamp=True)

        # Log some important data about the cross-validation session
        logger.logData([
            "\t \t \t \t Cross-validation session data",
            f"\n* k: {self.__k}",
            f"\n* Dimensionality of data: {self.__dimensionality}",
            f"\n* Total data: {self.__multitude}",
            f"\n* Validation-set size: {self.__multitude // self.__k}",
            f"\n* Train-set size: {self.__multitude - self.__multitude // self.__k}"
        ])

        # Create a model for each possible combination of the parameters
        models: List[MultiLayerPerceptron] = []
        for architecture in architectureValues:
            for epochs in epochValues:
                for learningRate in learningRateValues:
                    model: MultiLayerPerceptron = MultiLayerPerceptron(architecture, self.__dimensionality, epochs, learningRate)
                    models.append(model)

        for index, model in enumerate(models):
            messages: List[str] = []
            messages.append(f"Model: {index + 1}")
            messages.append("\n" + str(model))

            starTime = time.time()
            averageLoss = self.__crossValidate(model, index)

            endTime = time.time()

            # Print data
            messages.append(f"\nAverage loss: {averageLoss}")
            messages.append(f"\nDuration: {round(endTime - starTime)} seconds")

            logger.logData(messages)


    def __crossValidate(self, model: MultiLayerPerceptron, index: int) -> float:
        '''
            Trains and tests the given {model} using k-cross-validation. Returns the average loss
            in each k-fold.
        '''
        # Split the dataset {self.__X} and the targets {self.__y} into {self.__k} equal parts (chunks) 
        chunksX: Tuple[torch.Tensor] = self.__X.split(self.__k)
        chunksY: Tuple[torch.Tensor] = self.__y.split(self.__k)

        totalLoss: float = 0

        for i in range(self.__k):
            xTrain, xValidation = self.__getPartition(chunksX, i)
            yTrain, yValidation = self.__getPartition(chunksY, i)

            model.train(xTrain, yTrain)    
            loss = model.test(xValidation, yValidation)

            totalLoss += loss

        return round(totalLoss / self.__k, ndigits=2)


    def __getPartition(self, chunks: Tuple[torch.Tensor], i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
            Given the {chunks}, it returns the (trainSet, validationSet) at the given iteration {i} of the
            cross-validation technique as following: \n
            Let's say that, \n
            chunks = [C1, C2, C3, C4, C5, C6, C7, C8, C9, C10] and \n
            i = 7 \n
            Then: \n
            trainSet = the concatenation of C1, C2, C3, C4, C5, C6, C8, C9, C10\n
            validationSet = C7
        '''
        trainSet: torch.Tensor = []
        validationSet: torch.Tensor = []

        if i == 0:
            # In the first iteration, the first chunk is used as validation set, and the next k-1 as train set
            trainSet = torch.concat(chunks[1:])
            validationSet = chunks[0]
        elif i == self.__k - 1:
            # In the second iteration, the last chunk is used as validation set, and the previous k-1 as train set
            trainSet = torch.concat(chunks[0:self.__k-1])
            validationSet = chunks[self.__k-1]
        else:
            # The i-th chunk will be used for validation
            validationSet = chunks[i]

            # Concatenate all chunks leftside the validation chunk
            leftsideChunks: torch.Tensor = torch.concat(chunks[0:i])

            # Concatenate all chunks rightside the validation chunk
            rightsideChunks: torch.Tensor = torch.concat(chunks[i + 1:])

            # Combine leftside and rightside chunks into one tensor that will act as train set
            trainSet = torch.concat((leftsideChunks, rightsideChunks))

        return trainSet, validationSet