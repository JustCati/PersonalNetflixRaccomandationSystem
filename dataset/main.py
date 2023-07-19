from dataset.dataParser import *

from datetime import datetime
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem



def createTitleList(path, url, userAgentRotator):
    movies = []
    movies.append(datetime.today().strftime('%Y-%m-%d'))
    movies.extend(getCatalogueTitles(url, userAgentRotator))

    with open(path, "w") as f:
        f.write("\n".join(movies))
    
    return movies[1:]


def main():
    software_names = [SoftwareName.CHROME.value]
    operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value, OperatingSystem.MAC.value]
    userAgentRotator = UserAgent(software_names=software_names, operating_systems=operating_systems)
    
    for elem in ["serietv", "film"]:
        url = "https://movieplayer.it/" + elem + "/streaming/netflix/"
        pathCache = os.path.join(os.path.join(os.getcwd(), "cache", "{elem}.txt".format(elem=elem)))
        pathDF = os.path.join(os.path.join(os.getcwd(), "dataset"), "{elem}.parquet".format(elem=elem))

        movies = []
        if os.path.exists(pathCache):
            with open(pathCache, "r") as f:
                movies = f.readlines()
            date = movies[0].strip()
            if (datetime.today() - datetime.strptime(date, '%Y-%m-%d')).days < 7:
                movies = movies[1:]
            else:
                movies = createTitleList(pathCache, url, userAgentRotator)
        else:
            movies = createTitleList(pathCache, url, userAgentRotator)
        print("Trovati {n} {type}".format(n=len(movies), type=elem))

        dfMovies = getData(movies, userAgentRotator, pathDF)
        print(dfMovies.info())


if __name__ == "__main__":
    main()
