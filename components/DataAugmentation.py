from typing import List, Tuple

from PIL import Image, ImageEnhance

from customTypes.Pokemon import Pokemon
from customTypes.ProcessingType import ProcessingType

class DataAugmentation(object):
    '''
        This class is responsible for providing techniques that make the dataset larger. Due to the fact that
        the project deals with pokemon images, the techniques primarily involve image processing
    '''

    ######################################## PUBLIC METHODS ########################################

    def augment(self, data: List[Pokemon]) -> List[Pokemon]:
        '''
        '''
        newPokemons: List[Pokemon] = []
        for pokemon in data:
            augmentedImages: List[Tuple[Image.Image, ProcessingType]] = self.imageAugmentation(pokemon.image)
            
            for image, processingType in augmentedImages:
                augmentation: Pokemon = pokemon.getCopy()
                augmentation.name = f"{pokemon.name}-{processingType.name}"
                augmentation.setImage(image)

                newPokemons.append(augmentation)

        data.extend(newPokemons) 

        return data



    ######################################## PRIVATE METHODS ########################################
    def imageAugmentation(self, image: Image.Image) -> List[Tuple[Image.Image, ProcessingType]]: 
        '''
            Process the given {image} with the following ways and return the result images in a list: \n
            * Rotate 15" counter-clockwise
            * Rotate 15" clockwise
            * Increase brightness
            * Decrease brightness
        '''
        augmentedImages: List[Tuple[Image.Image, ProcessingType]] = []
        
        # Apply rotations
        augmentedImages.append((image.rotate(15), ProcessingType.CounterClockwiseRotation))
        augmentedImages.append((image.rotate(-15), ProcessingType.ClockwiseRotation))

        # Apply brightness adjustments
        augmentedImages.append((self.__adjustBrightness(image, 1.2), ProcessingType.IncreaseBrightness)) # Increase brightness
        augmentedImages.append((self.__adjustBrightness(image, 0.8), ProcessingType.DecreaseBrightness)) # Decrease brightness

        return augmentedImages



    def __adjustBrightness(self, image: Image.Image, brightnessFactor: float = 1.2) -> Image.Image:
        '''
            Creates a new image based on the given {image} by adding or removing brightness.
            The {brightnessFactor} controls how much color (brightness) will be added or removed.
            For {brightnessFactor} = 1, no change will be applied.
            For {brightnessFactor} < 1, color will be removed from the image
            For {brightnessFactor} > 1, color will be added to the image
        '''
        # Create an enhancer for the given {image}
        enhancer: ImageEnhance.Brightness = ImageEnhance.Brightness(image)

        # Adjusted image
        adjustedImage: Image.Image =enhancer.enhance(brightnessFactor)

        return adjustedImage