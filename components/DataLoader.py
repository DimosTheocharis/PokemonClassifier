import csv
from PIL import Image # type: ignore
from typing import List

from customTypes.Pokemon import Pokemon
from customTypes.PokemonType import PokemonType

class DataLoader():
    def __init__(self):
        self.__data: list[Pokemon] = []
    
    def readPokemonData(self, filePathForData: str, filePathForImages: str) -> List[Pokemon]:
        '''
            Reads both pokemon data such as name, types etc and pokemon images.
            Constructs a list of Pokemon instances containing all this information. The data
        '''
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


    def __readPokemonImage(self, name: str, filePath: str) -> Image:
        '''
            Reads the image of the pokemon with the given {name} and returns a NDArray of its pixels
        '''
        try:
            image = Image.open(f"{filePath}/{name}.png").convert("RGB")
            return image
        except:
            print(f"Could not load image data/images/{name}.png!")
        

