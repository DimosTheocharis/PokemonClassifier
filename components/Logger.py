from typing import List
from datetime import datetime


class Logger(object):
    '''
        This class is responsible for writing the outputs of the algorithms and other 
        important information at logger files. The data will be logged to .txt file, so the
        filePath should NOT contain any extension!
    '''
    def __init__(self, filePath: str, appendTimestamp: bool = True):
        if (appendTimestamp):
            self.__filePath: str = filePath + " " + datetime.now().strftime("%Y-%m-%d %H.%M") + ".txt"
        else:
            self.__filePath: str = filePath + ".txt"
        
    def logData(self, messages: List[str], printToConsole: bool = True) -> None:
        '''
            Logs data to the {self.__filePath}
            @messages: List[str] -> The data that i want to log into logger files.
            @printToConsole: bool -> Whether or not i want to log the data also to the console
        '''
        with open(self.__filePath, "a") as file:
            file.write("\n")
            file.writelines(messages)
            file.write("\n")

        if (printToConsole):
            print("")
            print(" ".join(messages))
            print("")

