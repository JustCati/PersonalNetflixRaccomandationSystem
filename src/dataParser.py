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
        except NoSuchElementException:
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
    except Exception:
        pass

    try:
        tramaSection = tramaSection.find_element(By.TAG_NAME, "div")
        driver.execute_script("arguments[0].scrollIntoView();", tramaSection.find_element(By.TAG_NAME, "a"));
        driver.execute_script("arguments[0].click();", tramaSection.find_element(By.TAG_NAME, "a"))
        ps = tramaSection.find_elements(By.TAG_NAME, "p")

        for p in ps:
            trama += p.text + "\n"
    except Exception:
        pass

    return trama.strip()


# def getElements(elem):
#     data = ""
#     try:
#         elements = elem.find_elements(By.TAG_NAME, "a")
#         for element in elements:
#             data += element.text + " "
#     except NoSuchElementException:
#         data = elem.text
#         print(data)
#     return data.strip()

def fetchInfo(mainDiv):
    keys = mainDiv.find_elements(By.TAG_NAME, "dt")
    elements = mainDiv.find_elements(By.TAG_NAME, "dd")

    data = dict.fromkeys(["Titolo originale", "Data di uscita", "Genere", "Anno", "Regia", "Attori", "Paese", "Durata", "Distribuzione"], None)

    for elem, i in zip(elements, keys):
        data[i.text] = elem.text

    data["Titolo"] = data.pop("Titolo originale", "")
    data["Data"] = data.pop("Data di uscita", "")
    
    if data["Data"] != "" and data["Data"] != None:
        data["Data"] = " ".join(data["Data"].split(" ", 3)[:-1])
    if data["Durata"] != "" and data["Durata"] != None:
        data["Durata"] = data["Durata"].split(" ")[0]
    return data


def getData(movies, rotator, path="movies.parquet"):
    col = ["id", "Titolo", "Data", "Genere", "Anno", "Regia", "Attori", "Paese", "Durata", "Distribuzione", "Trama"]

    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        df = pd.DataFrame([], columns=col)
        df = df.astype({
                "id" : "int64",
                "Titolo" : "string",
                "Data" : "string",
                "Genere" : "string",
                "Anno" : "string",
                "Regia" : "string",
                "Attori" : "string",
                "Paese" : "string",
                "Durata" : "string",
                "Distribuzione" : "string",
                "Trama" : "string"
                })
        df.to_parquet(path, index=False, engine="fastparquet")

    for movie in movies:
        if df[df.Titolo == movie].shape[0] <= 0:
            options = webdriver.ChromeOptions()
            # options.add_argument("--headless")
            options.add_argument("user-agent={userAgent}".format(userAgent=rotator.get_random_user_agent()))

            driver = webdriver.Chrome(options=options)
            driver.get(movie)
            
            try:
                mainDiv = driver.find_element(By.CLASS_NAME, "col-lg-8")
            except NoSuchElementException as e:
                movies.append(movie)
                continue

            id = (movie.split("_")[-1])[1:-1]

            titleElems = mainDiv.find_element(By.TAG_NAME, "header").find_element(By.TAG_NAME, "h1").text.split(" ")
            title = (" ".join(titleElems[:-1])).strip()

            i = 0
            trama = ""
            while trama == "" and i <= 5:
                trama = fetchTrama(driver, mainDiv)
                i += 1
            if trama == "":
                continue

            mainDiv = driver.find_element(By.XPATH, "//div[@class='scheda border p-3']").find_element(By.TAG_NAME, "dl")
            driver.execute_script("arguments[0].scrollIntoView();", mainDiv)

            data = fetchInfo(mainDiv)
            data["Titolo"] = title
            data["Trama"] = trama
            data["id"] = int(id)

            tempDF = pd.DataFrame([data], columns=col)
            tempDF = tempDF.astype({
                "id" : "int64",
                "Titolo" : "string",
                "Data" : "string",
                "Genere" : "string",
                "Anno" : "string",
                "Regia" : "string",
                "Attori" : "string",
                "Paese" : "string",
                "Durata" : "string",
                "Distribuzione" : "string",
                "Trama" : "string"})

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
