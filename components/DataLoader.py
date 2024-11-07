import csv
import os
from PIL import Image # type: ignore
from typing import List

from customTypes.Pokemon import Pokemon
from customTypes.PokemonType import PokemonType

class DataLoader():
    '''
        This class if responsible for providing methods that load data from files.
    '''
    def __init__(self):
        pass
    
    def readPokemonData(self, filePathForData: str, filePathForImages: str) -> List[Pokemon]:
        '''
            Reads both pokemon data such as name, types etc and pokemon images.
            Constructs a list of Pokemon instances containing all this information. 
        '''
        self.__data: list[Pokemon] = []
        with open(filePathForData, "r") as file:
            csv_reader = csv.reader(file)

            # Extract the first row of the data which is the names of the columns
            columnNames = csv_reader.__next__()

            for record in csv_reader:
                image = self.__readPokemonImage(record[0], filePathForImages)
                pokemon = Pokemon(record[0], PokemonType[record[1]], PokemonType[record[2]] if record[2] != "" else None)
                pokemon.setImage(image)
                self.__data.append(pokemon)

            return self.__data
        

    def storePokemonData(self, dataSet: List[Pokemon], filePath: str) -> None:
        '''
            Stores the data about the pokemon to a CSV file in the file-path location.
            For example, it can be used to store a specific train set, or test set. 
            The file should be CSV! \n
            The information that will be written for each pokemon will follow this pattern: \n
            name,type1,type2 \n

            if type2 does not exist, then an empty string will be placed there
        '''
        if filePath.split(".")[-1] != "csv":
            raise Exception(f"The file in the path {filePath} that you are trying to write, is not a CSV!")
        
        with open(filePath, "w") as file:
            lines: List[str] = []
            for pokemon in dataSet:
                lines.append(f"{pokemon.name},{pokemon.type1.name},{pokemon.type2.name if pokemon.type2 != None else ""}\n")

            file.writelines(lines)



    def __readPokemonImage(self, name: str, filePath: str) -> Image:
        '''
            Reads the image of the pokemon with the given {name} and returns a NDArray of its pixels
        '''
        try:
            image = Image.open(f"{filePath}/{name}.png").convert("RGB")
            return image
        except:
            print(f"Could not load image data/images/{name}.png!")

