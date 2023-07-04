from src.dataParser import *

from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem


def main():
    software_names = [SoftwareName.CHROME.value]
    operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value, OperatingSystem.MAC.value]
    userAgentRotator = UserAgent(software_names=software_names, operating_systems=operating_systems, limit=100)
    
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    
    for elem in ["film", "serietv"]:
        url = "https://movieplayer.it/" + elem + "/streaming/netflix/"
        filePath = os.path.join(os.getcwd(), "src", "dataList", "{elem}.txt".format(elem=elem))

        movies = loadLinks(url, options, userAgentRotator, filePath)
        print(elem.capitalize() + ": ", len(movies))
    
        dfMovies = getData(movies, options, userAgentRotator)
        dfMovies.to_csv(os.path.join(os.getcwd(), "{elem}.csv".format(elem=elem)), index=False)


#! ~ 4280 FILM TOTALI PRENDIBILI DA MOVIPLAYER.IT
if __name__ == "__main__":
    main()