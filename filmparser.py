import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By




def changePage(driver):
    nav = driver.find_element(By.CSS_SELECTOR, "[aria-label='Paginazione']").find_element(By.TAG_NAME, "ul")
    link = nav.find_elements(By.TAG_NAME, "li")[-1].find_element(By.TAG_NAME, "a").get_attribute("href")    
    if(link == ""):
        return False
    driver.get(link)
    return True


def getPageMoviesTitle(driver):
    movies = []
    movieList = driver.find_element(By.CLASS_NAME, "list--film").find_elements(By.TAG_NAME, "li")
    for movie in movieList:
        title = movie.find_element(By.TAG_NAME, "div").find_element(By.TAG_NAME, "h2").find_element(By.TAG_NAME, "a").get_attribute("href")
        movies.append(title)
    return movies


def getCatalogueTitles(driver):
    movies = []
    while True:
        movies.extend(getPageMoviesTitle(driver))
        if not changePage(driver):
            break
    with open("movies.txt", "w") as fp:
        fp.write("\n".join(movies))
    return movies


def loadMovies(driver):
    movies = []
    if not os.path.exists("movies.txt"):
        print("File not found, creating...")
        movies = getCatalogueTitles(driver)
        driver.close()
    else:
        print("File found, reading...")
        movies = open("movies.txt", "r").read().split("\n")
    return movies



def main():
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")

    driver = webdriver.Chrome(options=options)
    driver.get("https://movieplayer.it/film/streaming/netflix/")

    movies = loadMovies(driver)
    print("Movies: ", len(movies))
        
    
    
    
    
    
    
    

#! ~ 4280 FILM TOTALI PRENDIBILI DA MOVIPLAYER.IT
if __name__ == "__main__":
    main()