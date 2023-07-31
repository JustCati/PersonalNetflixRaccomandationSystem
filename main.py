import os
import random
import argparse
import numpy as np
import pandas as pd

from raccomend.distances import *
from raccomend.embeddings import *

from dataset.dataScraper import getDataset
from dataset.raccomenderDataset import getUtilityMatrix

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error




def main():
    parser = argparse.ArgumentParser(description="Raccomender")
    parser.add_argument("-q", "--qualitative", action="store_true", default=None, help="Show naive qualitative raccomendation")
    parser.add_argument("-c", "--count", type=int, default=1, help="Set the number of ratings to use for training for each user")
    parser.add_argument("-l", "--linear", action="store_true", help="Use linear regression for raccomendation")
    args = parser.parse_args()


    #* ---------- Load Dataset ------------
    #* Load Movies metadata Dataset
    if os.path.exists("netflix.parquet"):
        movies = pd.read_parquet("netflix.parquet")
    else:
        movies = getDataset(update=False)              #! DEBUG, change with update=True
        movies.to_parquet("netflix.parquet")

    #* Load Embeddings Dataset
    if os.path.exists("embeddings.parquet"):
        embeddings = pd.read_parquet("embeddings.parquet")
    else:
        embeddings = pd.DataFrame({"id" : movies.id, "Tipologia" : movies.Tipologia, "Titolo" : movies.Titolo}, columns=["id", "Tipologia", "Titolo"])
        embeddings["Embeddings_Trama"] = [np.full((1024,), np.inf, dtype=np.float32) for _ in range(len(movies))]
    #* ------------------------------------


    #* ---------- Get Embeddings ------------
    toUpdate = False
    if (embeddings.Embeddings_Trama.values[0] == [np.full((1024,), np.inf, dtype=np.float32) for _ in range(len(movies))]).all():
        embeddings = getEmbeddingsTrama_E5_LargeV2(embeddings, movies, shuffle=True)
        toUpdate = True


    if toUpdate or "allEmbeddings" not in embeddings.columns:
        embeddings = getFeatureGenere(embeddings, movies)
        embeddings = getFeatureAttori(embeddings, movies, colName="Regia")
        embeddings = getFeatureAttori(embeddings, movies, colName="Attori")
        embeddings = getFeatureTipologia(embeddings, movies)

        embeddings["allEmbeddings"] = embeddings.apply(lambda x: np.concatenate((x["Embeddings_Trama"], x["Embeddings_Genere"], x["Embeddings_Regia"], x["Embeddings_Attori"], x["Embeddings_Tipologia"]), dtype=np.float32), axis=1)
        embeddings = embeddings.drop(columns=["Embeddings_Genere", "Embeddings_Regia", "Embeddings_Attori", "Embeddings_Tipologia"])
        embeddings.allEmbeddings = embeddings.apply(lambda x: StandardScaler().fit_transform(x["allEmbeddings"].reshape(-1, 1)).reshape(-1), axis=1)

        pca = PCA(n_components=1024)
        data = embeddings.allEmbeddings.values
        data = np.array([np.array(elem) for elem in data])

        pca.fit(data)
        data = pca.transform(data)
        embeddings.allEmbeddings = data.tolist()
        embeddings.to_parquet("embeddings.parquet")
        
        embeddings = embeddings.drop(columns=["Embeddings_Trama"])
    #* --------------------------------------------


    #* ---------- Naive Raccomender based on similarity --------
    if args.qualitative is not None:
        while((title := input("Inserisci il titolo del film: ").strip()) not in movies.Titolo.values):
            print("Film non trovato")

        print("Film simili con cosine come distanza:")
        print(getMostSimilarCosine(movies, embeddings, title))

        print("\nFilm simili con euclidean come distanza:")
        print(getMostSimilarEuclidean(movies, embeddings, title))
    #* ---------------------------------------------------------


    #* ---------- Utility Matrix -------------
    if not os.path.exists("utilitymatrix.parquet"):
        utilityMatrix = getUtilityMatrix(movies)
        utilityMatrix.to_parquet("utilitymatrix.parquet")
    else:
        utilityMatrix = pd.read_parquet("utilitymatrix.parquet")
    #* ---------------------------------------

    #* ---------- Dataset Creation ------------
    columns = utilityMatrix.columns.tolist()
    random.Random(42).shuffle(columns)          #! DEBUG, change with random.Random().shuffle(columns) for random order
    columns = columns[:args.count]
    remaining = [elem for elem in utilityMatrix.columns.tolist() if elem not in columns]

    validUsers = None
    for col in columns:
        validUsers = utilityMatrix[utilityMatrix[col] != 0] if validUsers is None else validUsers[validUsers[col] != 0]

    train = validUsers[columns]
    test = validUsers[remaining]
    #* ---------------------------------------


    #* ---------- Linear Regressor -------------
    if args.linear:
        preds = np.array([])
        ratings = np.array([])
        model = LinearRegression(n_jobs=-1)

        for ((_, rowX), (_, rowY)) in zip(train.iterrows(), test.iterrows()):
            dfTrain = pd.DataFrame({"Titolo" : train.columns, "Embeddings" : [embeddings[embeddings.Titolo == elem].allEmbeddings.values[0] for elem in train.columns], "Rating" : rowX.values})
            dfTest = pd.DataFrame({"Titolo" : test.columns, "Embeddings" : [embeddings[embeddings.Titolo == rem].allEmbeddings.values[0] for rem in remaining], "Rating" : rowY.values})
            dfTest = dfTest[dfTest.Rating != 0]

            model.fit(dfTrain.Embeddings.tolist(), dfTrain.Rating.tolist())
            pred = model.predict(dfTest.Embeddings.tolist())
            
            preds = np.append(preds, pred)
            ratings = np.append(ratings, dfTest.Rating.values)

        print()
        print("Linear Regression: ")
        print(f"RMSE: {mean_squared_error(ratings, preds, squared=True)}")
    #* ----------------------------------------



if __name__ == "__main__":
    main()
