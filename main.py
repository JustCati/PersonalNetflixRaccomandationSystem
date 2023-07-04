from src.dataParser import *

from selenium import webdriver



def main():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    
    for elem in ["film", "serietv"]:

        driver = webdriver.Chrome(options=options)
        driver.get("https://movieplayer.it/" + elem + "/streaming/netflix/")

        filePath = os.path.join(os.getcwd(), "src", "dataList", "{elem}.txt".format(elem=elem))

        movies = loadLinks(driver, filePath)
        print(elem.capitalize + ": ", len(movies))
    
    # dfMovies = getData(driver, movies)



#! ~ 4280 FILM TOTALI PRENDIBILI DA MOVIPLAYER.IT
if __name__ == "__main__":
    main()