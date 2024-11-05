from Scenarios import Scenarios
from customTypes.Settings import FeatureReductionSettings

import time

import numpy as np

startTime = time.time()
Scenarios.KNNScenario(3, FeatureReductionSettings(newWidth=30, newHeight=30))
endTime = time.time()
print(f"Total time: {round(endTime - startTime)} seconds")

startTime = time.time()
Scenarios.KNNScenario(3, FeatureReductionSettings(newWidth=30, newHeight=30), augmentData=True)
endTime = time.time()
print(f"Total time: {round(endTime - startTime)} seconds")

startTime = time.time()
Scenarios.NCCScenario(FeatureReductionSettings(newWidth=30, newHeight=30))
endTime = time.time()
print(f"Total time: {round(endTime - startTime)} seconds")

startTime = time.time()
Scenarios.NCCScenario(FeatureReductionSettings(newWidth=30, newHeight=30), augmentData=True)
endTime = time.time()
print(f"Total time: {round(endTime - startTime)} seconds")