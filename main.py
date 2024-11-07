from Scenarios import Scenarios
from customTypes.Settings import FeatureReductionSettings


# Scenarios.MLPScenario(FeatureReductionSettings(newWidth=30, newHeight=30))

# Scenarios.MLPScenario(FeatureReductionSettings(newWidth=30, newHeight=30), augmentData=True)

data = Scenarios.readAndSplitData(readSavedData=True)
testSet = data[1]

from components.DataAugmentation import DataAugmentation

da = DataAugmentation()

kyogre = [pokemon for pokemon in testSet if pokemon.name == "kyogre"][0]

kyogre.image.show()

for image, processingType in da.imageAugmentation(kyogre.image):
    print(processingType)
    image.show()


# Scenarios.KNNScenario(3, FeatureReductionSettings(newWidth=30, newHeight=30))
# Scenarios.KNNScenario(3, FeatureReductionSettings(newWidth=30, newHeight=30), augmentData=True)
# Scenarios.NCCScenario(FeatureReductionSettings(newWidth=30, newHeight=30))
# Scenarios.NCCScenario(FeatureReductionSettings(newWidth=30, newHeight=30), augmentData=True)