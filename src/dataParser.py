import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By

from selenium.webdriver.common.keys import Keys


def changePage(url, rotator):
    options = webdriver.ChromeOptions()
    options.add_argument("user-agent={userAgent}".format(userAgent=rotator.get_random_user_agent()))
    
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    nav = driver.find_element(By.CSS_SELECTOR, "[aria-label='Paginazione']").find_element(By.TAG_NAME, "ul")
    link = nav.find_elements(By.TAG_NAME, "li")[-1].find_element(By.TAG_NAME, "a").get_attribute("href")    
    return link


def getPageMoviesTitle(url, rotator):
    options = webdriver.ChromeOptions()
    options.add_argument("user-agent={userAgent}".format(userAgent=rotator.get_random_user_agent()))
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    
    movies = []
    movieList = driver.find_element(By.CLASS_NAME, "list--film").find_elements(By.TAG_NAME, "li")
    for movie in movieList:
        title = movie.find_element(By.TAG_NAME, "div").find_element(By.TAG_NAME, "h2").find_element(By.TAG_NAME, "a").get_attribute("href")
        movies.append(title)
    driver.close()
    return movies


def getCatalogueTitles(url, rotator, path="movies.txt"):
    first = True
    movies = []
    
    while True:
        if first:
            first = False
            movies.extend(getPageMoviesTitle(url, rotator))
            continue
        
        url = changePage(url, rotator)
        if url == "":
            break
        movies.extend(getPageMoviesTitle(url, rotator))
        
    with open(path, "w") as fp:
        fp.write("\n".join(movies))
    return movies


def loadLinks(url, rotator, path="movies.txt"):
    movies = []
    if not os.path.exists(path):
        print("File not found, creating...") 
        movies = getCatalogueTitles(url, rotator, path)
    else:
        print("File found, reading...")
        movies = open(path, "r").read().split("\n")
    return movies


def fetchTrama(driver, mainDiv):
    trama = ""
    
    try:
        tramaSection = mainDiv.find_element(By.CLASS_NAME, "section-scheda--trama")
        trama += tramaSection.find_element(By.TAG_NAME, "p").text + "\n"
    except:
        pass

    try:
        tramaSection = tramaSection.find_element(By.TAG_NAME, "div")
        driver.execute_script("arguments[0].click();", tramaSection.find_element(By.TAG_NAME, "a"))
        ps = tramaSection.find_elements(By.TAG_NAME, "p")
        
        for p in ps:
            trama += p.text + "\n"
    except: 
        pass
    
    return trama.strip()

def saveRemaining(remaining, path="movies.txt"):
    with open(path, "w") as fp:
        fp.write("\n".join(remaining))


def getData(movies, rotator, path="movies.csv"):
    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        df = pd.DataFrame(columns=["title", "trama"])
        df.to_parquet(path, index=False, engine="fastparquet")
    
    remaining = set(movies)
    for movie in movies:
        options = webdriver.ChromeOptions()
        options.add_argument("user-agent={userAgent}".format(userAgent=rotator.get_random_user_agent()))
        driver = webdriver.Chrome(options=options)
        driver.get(movie)
        
        mainDiv = driver.find_element(By.CLASS_NAME, "col-lg-8")
        title = mainDiv.find_element(By.TAG_NAME, "header").find_element(By.TAG_NAME, "h1").text

        i = 0
        trama = ""
        while trama == "" and i < 5:
            html = driver.find_element(By.TAG_NAME, 'html')
            html.send_keys(Keys.PAGE_DOWN)
            time.sleep(1)
            
            trama = fetchTrama(driver, mainDiv)
            i += 1
        if trama == "":
            continue
        
        tempDF = pd.DataFrame([[title, trama]], columns=["title", "trama"])
        if df[df.title == title].shape[0] <= 0:
            df = pd.concat([df, tempDF], ignore_index=True)
            tempDF.to_parquet(path, index=False, engine="fastparquet", append=True)
        
        driver.close()
        remaining.remove(movie)
        saveRemaining(remaining, os.path.join("list", path.replace(".parquet", ".txt")))

    return df
