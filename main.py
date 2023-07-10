from src.dataParser import *

from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem


def main():
    software_names = [SoftwareName.CHROME.value]
    operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value, OperatingSystem.MAC.value]
    userAgentRotator = UserAgent(software_names=software_names, operating_systems=operating_systems)
    
    for elem in ["film", "serietv"]:
        url = "https://movieplayer.it/" + elem + "/streaming/netflix/"
        dfPath = os.path.join(os.path.join(os.getcwd(), "data", "{elem}.parquet".format(elem=elem)))

        movies = getCatalogueTitles(url, userAgentRotator)
        print("Trovati {n} {type}".format(n=len(movies), type=elem))

        dfMovies = getData(movies, userAgentRotator, dfPath)
        print(dfMovies.info())


if __name__ == "__main__":
    main()
