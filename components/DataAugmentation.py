from typing import List, Tuple
import random

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
            augmentedImages: List[Tuple[Image.Image, List[ProcessingType]]] = self.imageAugmentation(pokemon.image)
            
            for image, processingTypes in augmentedImages:
                augmentation: Pokemon = pokemon.getCopy()
                for processingType in processingTypes:
                    augmentation.name = f"{augmentation.name}-{processingType.name}"
                augmentation.setImage(image)

                newPokemons.append(augmentation)

        data.extend(newPokemons) 

        return data



    ######################################## PRIVATE METHODS ########################################

    def imageAugmentationOld(self, image: Image.Image) -> List[Tuple[Image.Image, ProcessingType]]: 
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
    
    def imageAugmentation(self, image: Image.Image) -> List[Tuple[Image.Image, List[ProcessingType]]]: 
        '''
            Process the given {image} with the following ways: \n
            * Rotate 15" counter-clockwise
            * Rotate 15" clockwise
            * Increase brightness
            * Decrease brightness \n

            and return the result images in a list
        '''
        augmentedImages: List[Tuple[Image.Image, List[ProcessingType]]] = []

        for i in range(15):
            newImage: Image.Image = image.copy()
            processingTypes: List[ProcessingType] = []

            # Apply rotations
            if (random.random() >= 0.5):
                newImage = newImage.rotate(30)
                processingTypes.append(ProcessingType.CounterClockwiseRotation)
            else:
                if (random.random() >= 0.5): 
                    newImage = newImage.rotate(-30)
                    processingTypes.append(ProcessingType.ClockwiseRotation)

            # Apply brightness adjustments
            if (random.random() >= 0.5): 
                newImage = self.__adjustBrightness(newImage, 1.2)
                processingTypes.append(ProcessingType.IncreaseBrightness)
            else:
                if (random.random() >= 0.5): 
                    newImage = self.__adjustBrightness(newImage, 0.8)
                    processingTypes.append(ProcessingType.DecreaseBrightness)

            # Crop
            if (random.random() >= 0.8):
                r = random.random()
                if (r <= 0.2):
                    newImage = self.cropImageCenter(newImage)
                    processingTypes.append(ProcessingType.CropCenter)
                elif (r <= 0.4):
                    newImage = self.cropImageTopLeftCorner(newImage)
                    processingTypes.append(ProcessingType.CropTopLeftCorner)
                elif (r <= 0.6):
                    newImage = self.cropImageTopRightCorner(newImage)
                    processingTypes.append(ProcessingType.CropTopRightCorner)
                elif (r <= 0.8):
                    newImage = self.cropImageBottomRightCorner(newImage)
                    processingTypes.append(ProcessingType.CropBottomRightCorner)
                else:
                    newImage = self.cropImageBottomLeftCorner(newImage)
                    processingTypes.append(ProcessingType.CropBottomLeftCorner)


            if (len(processingTypes) > 0 and any(processingTypes == processing for processing in [item[1] for item in augmentedImages])) == False:
                augmentedImages.append((newImage, processingTypes))

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
    


    def cropImageCenter(self, image: Image.Image) -> Image.Image:
        '''
        '''
        width: int = image.width
        height: int = image.height
        cropBox: Tuple[int] = (round(width * 0.25), round(height * 0.25), round(width * 0.75), round(height * 0.75))

        return self.__cropImage(image, cropBox)
    

    def cropImageTopLeftCorner(self, image: Image.Image) -> Image.Image:
        '''
        '''
        width: int = image.width
        height: int = image.height
        cropBox: Tuple[int] = (round(width * 0), round(height * 0), round(width * 0.5), round(height * 0.5))

        return self.__cropImage(image, cropBox)
    

    def cropImageTopRightCorner(self, image: Image.Image) -> Image.Image:
        '''
        '''
        width: int = image.width
        height: int = image.height
        cropBox: Tuple[int] = (round(width * 0.5), round(height * 0), round(width * 1), round(height * 0.5))

        return self.__cropImage(image, cropBox)
    

    def cropImageBottomRightCorner(self, image: Image.Image) -> Image.Image:
        '''
        '''
        width: int = image.width
        height: int = image.height
        cropBox: Tuple[int] = (round(width * 0.5), round(height * 0.5), round(width * 1), round(height * 1))

        return self.__cropImage(image, cropBox)


    def cropImageBottomLeftCorner(self, image: Image.Image) -> Image.Image:
        '''
        '''
        width: int = image.width
        height: int = image.height
        cropBox: Tuple[int] = (round(width * 0), round(height * 0.5), round(width * 0.5), round(height * 1))

        return self.__cropImage(image, cropBox)

    def __cropImage(self, image: Image.Image, cropBox: Tuple[int]) -> Image.Image:
        '''
        '''
        # Cropped image
        croppedImage: Image.Image = image.crop(cropBox)

        return croppedImage.resize((image.width, image.height))


