import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import matplotlib.pyplot as plt

from customTypes.NeuralNetworkTypes import LayerType, ActivationFunctionType

class MultiLayerPerceptron(nn.Module):
    
    def __init__(self, 
            layers: List[Tuple[LayerType, int, ActivationFunctionType]] = [
                (LayerType.Linear, 512, ActivationFunctionType.ReLu),
                (LayerType.Linear, 256, ActivationFunctionType.ReLu),
                (LayerType.Linear, 128, ActivationFunctionType.ReLu),
                (LayerType.Linear, 18, ActivationFunctionType.Sigmoid),
            ],
            dimensionality: int = 2700,
            epochs: int = 100,
            learningRate: float = 0.01
        ):
        super().__init__()

        # Store network parameters
        self.__dimensionality: int = dimensionality
        self.__epochs: int = epochs
        self.__learningRate: float = learningRate


        self.__model: nn.Sequential = nn.Sequential()
        inFeatures: int = self.__dimensionality
        for layer in layers:
            if layer[0] == LayerType.Linear:
                self.__model.append(nn.Linear(inFeatures, layer[1], bias=True, dtype=torch.float32))

            if layer[2] == ActivationFunctionType.ReLu:
                self.__model.append(nn.ReLU())
            elif layer[2] == ActivationFunctionType.Sigmoid:
                self.__model.append(nn.Sigmoid())

            inFeatures = layer[1]


    def train(
            self, 
            xTrain: torch.Tensor, 
            yTrain: torch.Tensor, 
            calculateValidationLoss: bool = False,
            xValidation: torch.Tensor | None = None, 
            yValidation: torch.Tensor | None = None
        ):
        # I will calculate the loss of my network's prediction with Binary Cross-Entropy Loss
        criterion: torch.nn.BCELoss = torch.nn.BCELoss()

        # Set Adam as optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.__learningRate)

        # Store the loss of the network running in the train and validation sets, for each epoch 
        trainLosses: List[float] = []
        validationLosses: List[float] = []

        for i in range(self.__epochs):
            # Feed forward the xTrain data to the network
            yPredictionForTrainSet: torch.Tensor = self(xTrain)

            # Compute the loss of the network running in the train set
            trainLoss: torch.Tensor = criterion(yPredictionForTrainSet, yTrain)
            trainLosses.append(trainLoss.item())
            
            if (calculateValidationLoss):
                # Feed forward the xValidation data to the network
                yPredictionForValidationSet: torch.Tensor = self(xValidation)

                # Compute the loss of the network running in the validation set
                validationLoss: torch.Tensor = criterion(yPredictionForValidationSet, yValidation)
                validationLosses.append(validationLoss.item())

            # Set the gradients to zero
            self.zero_grad()

            # Back propagation
            trainLoss.backward()
            optimizer.step()

        if (calculateValidationLoss):
            # Make a graph with the losses
            plt.plot(trainLosses, color="r" ,label="Training Loss")
            plt.plot(validationLosses, color="g", label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training & Validation Loss over Epochs")
            plt.legend()
            plt.show()


    def test(self, xTest: torch.Tensor, yTest: torch.Tensor) -> float:
        '''
            Feeds the {xTest} data into the trained network and calculates the loss by comparing the target {yTest} with the
            result of the network. \n
            @returns: the loss
        '''
        # Turn-off back propagation
        with torch.no_grad():
            yEvaluation = self(xTest)
            criterion: torch.nn.BCELoss = torch.nn.BCELoss()

            loss: torch.Tensor = criterion(yEvaluation, yTest)
            return loss.item()

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.__model(x)

        return output
    

    def __str__(self):
        return super().__str__() + f"\nEpochs = {self.__epochs}" + f"\nLearning rate = {self.__learningRate}" 