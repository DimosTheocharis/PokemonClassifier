import csv
from PIL import Image # type: ignore
import numpy as np
from numpy._typing import NDArray

from customTypes.Pokemon import Pokemon
from customTypes.PokemonType import PokemonType

class DataLoader():
    def __init__(self):
        self.data: list[Pokemon] = []
    
    def readPokemonData(self, filePath: str) -> list[Pokemon]:
        '''
            Reads both pokemon data such as name, types etc and pokemon images.
            Constructs a list of Pokemon instances containing all this information
        '''
        with open(filePath, "r") as file:
            csv_reader = csv.reader(file)

            # Extract the first row of the data which is the names of the columns
            columnNames = csv_reader.__next__()

            for record in csv_reader:
                image = self.__readPokemonImage(record[0])
                pokemon = Pokemon(record[0], PokemonType[record[1]], PokemonType[record[2]] if record[2] != "" else None)
                pokemon.setImage(image)
                self.data.append(pokemon)

            return self.data
        



    def readPokemonImages(self):
        img1 = Image.open("data/images/bulbasaur.png").convert("RGB")

        img2 = Image.open("data/images/ivysaur.png").convert("RGB")

        img3 = Image.open("data/images/charmander.png").convert("RGB")

        pixels1: NDArray[np.integer] = np.array(img1)
        pixels2: NDArray[any] = np.array(img2)
        pixels3: NDArray[any] = np.array(img3)

        diff1 = np.linalg.norm(pixels1 - pixels2)
        diff2 = np.linalg.norm(pixels1 - pixels3)

        print(f"Bulbasaur vs Ivysaur => {diff1}")
        print(f"Bulbasaur vs Charmander => {diff2}")



    def __readPokemonImage(self, name: str) -> NDArray[np.integer]:
        '''
            Reads the image of the pokemon with the given {name} and returns a NDArray of its pixels
        '''
        try:
            image = Image.open(f"data/pokemonImages/{name}.png").convert("RGB")
            return np.array(image)
        except:
            print(f"Could not load image data/images/{name}.png!")
        

