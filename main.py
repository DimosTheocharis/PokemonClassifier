from Scenarios import Scenarios
from customTypes.Settings import FeatureReductionSettings


#Scenarios.MLPScenario(FeatureReductionSettings(newWidth=30, newHeight=30))

# Scenarios.MLPScenario(FeatureReductionSettings(newWidth=30, newHeight=30), augmentData=True)

# Scenarios.crossValidationScenario(FeatureReductionSettings(newWidth=30, newHeight=30), augmentData=True, readSavedData=True)



Scenarios.KNNScenario(1, FeatureReductionSettings(newWidth=30, newHeight=30))
Scenarios.KNNScenario(3, FeatureReductionSettings(newWidth=30, newHeight=30))
Scenarios.NCCScenario(FeatureReductionSettings(newWidth=30, newHeight=30))