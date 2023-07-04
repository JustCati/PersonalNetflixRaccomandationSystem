import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By



def changePage(url, options, rotator):
    options.add_argument("user-agent={userAgent}".format(userAgent=rotator.get_random_user_agent()))
    
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    nav = driver.find_element(By.CSS_SELECTOR, "[aria-label='Paginazione']").find_element(By.TAG_NAME, "ul")
    link = nav.find_elements(By.TAG_NAME, "li")[-1].find_element(By.TAG_NAME, "a").get_attribute("href")    
    return link


def getPageMoviesTitle(url, options, rotator):
    options.add_argument("user-agent={userAgent}".format(userAgent=rotator.get_random_user_agent()))
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    
    movies = []
    movieList = driver.find_element(By.CLASS_NAME, "list--film").find_elements(By.TAG_NAME, "li")
    for movie in movieList:
        title = movie.find_element(By.TAG_NAME, "div").find_element(By.TAG_NAME, "h2").find_element(By.TAG_NAME, "a").get_attribute("href")
        movies.append(title)
        print(driver.execute_script("return navigator.userAgent"))
    driver.close()
    return movies


def getCatalogueTitles(url, options, rotator, path="movies.txt"):
    first = True
    movies = []
    
    while True:
        if first:
            first = False
            movies.extend(getPageMoviesTitle(url, options, rotator))
            continue
        
        url = changePage(url, options, rotator)
        if url == "":
            break
        movies.extend(getPageMoviesTitle(url, options, rotator))
        
    with open(path, "w") as fp:
        fp.write("\n".join(movies))
    return movies


def loadLinks(url, options, rotator, path="movies.txt"):
    movies = []
    if not os.path.exists(path):
        print("File not found, creating...") 
        movies = getCatalogueTitles(url, options, rotator, path)
    else:
        print("File found, reading...")
        movies = open(path, "r").read().split("\n")
    return movies


def getData(driver, movies):
    df = pd.DataFrame(columns=["title", "trama"])
    
    for movie in movies:
        driver.get(movie)
        mainDiv = driver.find_element(By.CLASS_NAME, "col-lg-8")
        
        title = mainDiv.find_element(By.TAG_NAME, "header").find_element(By.TAG_NAME, "h1").text

        trama = ""
        try:
            tramaSection = mainDiv.find_element(By.CLASS_NAME, "section-scheda--trama")
            trama += tramaSection.find_element(By.TAG_NAME, "p").text + "\n"
        except:
            continue
        
        try:
            tramaSection = tramaSection.find_element(By.TAG_NAME, "div")
            driver.execute_script("arguments[0].click();", tramaSection.find_element(By.TAG_NAME, "a"))
            ps = tramaSection.find_elements(By.TAG_NAME, "p")
            
            for p in ps:
                trama += p.text + "\n"
        except:
            pass
        
        tempDF = pd.DataFrame([[title, trama]], columns=["title", "trama"])
        df = pd.concat([df, tempDF], ignore_index=True)
    return df
