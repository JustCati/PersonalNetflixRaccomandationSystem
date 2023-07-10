import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException


def changePage(url, rotator):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("user-agent={userAgent}".format(userAgent=rotator.get_random_user_agent()))
    
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    
    nav = driver.find_element(By.CSS_SELECTOR, "[aria-label='Paginazione']").find_element(By.TAG_NAME, "ul")
    link = nav.find_elements(By.TAG_NAME, "li")[-1].find_element(By.TAG_NAME, "a").get_attribute("href")    
    return link


def getPageMoviesTitle(url, rotator):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("user-agent={userAgent}".format(userAgent=rotator.get_random_user_agent()))
    
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    
    movies = []
    movieList = driver.find_element(By.CLASS_NAME, "list--film").find_elements(By.TAG_NAME, "li")
    for movie in movieList:
        try:
            driver.execute_script("arguments[0].scrollIntoView();", movie)
            title = movie.find_element(By.TAG_NAME, "div").find_element(By.TAG_NAME, "h2").find_element(By.TAG_NAME, "a").get_attribute("href")
            movies.append(title)
        except NoSuchElementException as e:
            movieList.append(movie)
            continue
    driver.close()
    return movies


def getCatalogueTitles(url, rotator):
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
    return movies


def fetchTrama(driver, mainDiv):
    trama = ""
    
    try:
        tramaSection = mainDiv.find_element(By.CLASS_NAME, "section-scheda--trama")
        driver.execute_script("arguments[0].scrollIntoView();", tramaSection)
        
        trama += tramaSection.find_element(By.TAG_NAME, "p").text + "\n"
    except Exception as e:
        pass

    try:
        tramaSection = tramaSection.find_element(By.TAG_NAME, "div")
        driver.execute_script("arguments[0].scrollIntoView();", tramaSection.find_element(By.TAG_NAME, "a"));
        driver.execute_script("arguments[0].click();", tramaSection.find_element(By.TAG_NAME, "a"))
        ps = tramaSection.find_elements(By.TAG_NAME, "p")
        
        for p in ps:
            trama += p.text + "\n"
    except Exception as e: 
        pass
    
    return trama.strip()


def fetchTitle(mainDiv):
    #TODO ADD PARSING TO REMOVE DATA FROM TITLE
    return mainDiv.find_element(By.TAG_NAME, "header").find_element(By.TAG_NAME, "h1").text



def getData(movies, rotator, path="movies.parquet"):
    col = ["id", "title", "trama", "anno", "cast", "regia", "genere", "durata", "data_uscita" "durata"]
    
    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        df = pd.DataFrame(columns=col[:4])
        df.to_parquet(path, index=False, engine="fastparquet")

    for movie in movies:
        if df[df.title == movie].shape[0] <= 0:
            options = webdriver.ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("user-agent={userAgent}".format(userAgent=rotator.get_random_user_agent()))
            
            driver = webdriver.Chrome(options=options)
            driver.get(movie)
            
            id = (movie.split("_")[-1])[1:-1]
            
            try:
                mainDiv = driver.find_element(By.CLASS_NAME, "col-lg-8")
            except NoSuchElementException as e:
                movies.append(movie)
                continue

            titleElems = fetchTitle(mainDiv).split(" ")
            title = (" ".join(titleElems[:-1])).strip()
            anno = titleElems[-1].strip()

            i = 0
            trama = ""
            while trama == "" and i <= 5:
                trama = fetchTrama(driver, mainDiv)
                i += 1
            if trama == "":
                continue
            
            print(id, title, anno, ": ", trama) #! DEBUG DEBUG DEBUG
             
            tempDF = pd.DataFrame([[id, title, trama, anno]], columns=col[:4])
            df = pd.concat([df, tempDF], ignore_index=True)
            tempDF.to_parquet(path, index=False, engine="fastparquet", append=True)
            
            driver.close()
        else:
            pass
            #TODO: CICLA TUTTE LE COLONNE E CONTROLLA SE VUOTE
            #TODO: SE VUOTE, CERCA NELLA PAGINA E AGGIORNA ALMENO UNA VOLTA
            # for col in df.columns:
            #     if col == "title":
            #         continue
    return df
