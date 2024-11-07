import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import matplotlib.pyplot as plt

from customTypes.NeuralNetworkTypes import LayerType, ActivationFunctionType

class MultiLayerPerceptron(nn.Module):
    
    def __init__(self, 
            numberOfLayers: int = 4, 
            neuronsPerLayer: List[int] = [512, 256, 128, 18], 
            layers: List[Tuple[str, int, str]] = [
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
        self.__numberOfLayers: int = numberOfLayers
        self.__neuronsPerLayer: int = neuronsPerLayer
        self.__dimensionality: int = dimensionality
        self.__epochs: int = epochs
        self.__learningRate: float = learningRate

        # The user should give the number of neurons for every layer.
        if (numberOfLayers != len(neuronsPerLayer)):
            raise Exception(f"The network has {self.__numberOfLayers} layers but you provided neurons for {self.__neuronsPerLayer} layers.")


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



    def train(self, xTrain: torch.Tensor, yTrain: torch.Tensor):
        # I will calculate the loss of my network's prediction with Binary Cross-Entropy Loss
        criterion: torch.nn.BCELoss = torch.nn.BCELoss()

        # Set Adam as optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.__learningRate)

        # Store the loss for each epoch
        losses: List[float] = []

        for i in range(self.__epochs):
            # Feed forward the xTrain data to the network
            yPrediction: torch.Tensor = self(xTrain)

            # Compute the loss
            loss: torch.Tensor = criterion(yPrediction, yTrain)
            losses.append(loss.item())

            # Set the gradients to zero
            self.zero_grad()

            # Back propagation
            loss.backward()
            optimizer.step()


        # Make a graph with the losses
        plt.plot(losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.legend()
        plt.show()


    def test(self, xTest: torch.Tensor, yTest: torch.Tensor):
        # Turn-off back propagation
        with torch.no_grad():
            print("x-set:")
            print(xTest)
            print("y-set:")
            print(yTest)
            yEvaluation = self(xTest)
            print("evaluation:")
            print(yEvaluation)
            criterion: torch.nn.BCELoss = torch.nn.BCELoss()

            loss = criterion(yEvaluation, yTest)
            print("loss:")
            print(loss)




    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.__model(x)

        return output