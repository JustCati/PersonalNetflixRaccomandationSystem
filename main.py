import os
import random
import argparse
import numpy as np
import pandas as pd

from raccomend.distances import *
from raccomend.embeddings import *
from raccomend.predict import predict, predictWithUser

from sklearn.preprocessing import StandardScaler

from dataset.dataScraper import getDataset
from dataset.raccomenderDataset import getUtilityMatrix






def main():
    parser = argparse.ArgumentParser(description="Raccomender")
    parser.add_argument("-q", "--qualitative", action="store_true", default=False, help="Show naive qualitative raccomendation")
    parser.add_argument("-u", "--user", action="store_true", default=False, help="Show results with user profile similarity")
    parser.add_argument("-c", "--count", type=int, default=5, help="Set the number of ratings to use for training for each user")
    parser.add_argument("-a", "--algorithm", type=str, default="", help="Set the algorithm to use for raccomendation")
    parser.add_argument("-r", "--round", action="store_true", default=False, help="Round the predicted ratings")
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

    if toUpdate:
        embeddings = getFeatureGenere(embeddings, movies)
        embeddings = getFeatureAttori(embeddings, movies, colName="Regia")
        embeddings = getFeatureAttori(embeddings, movies, colName="Attori")
        embeddings = getFeatureTipologia(embeddings, movies)

        embeddings["allEmbeddings"] = embeddings.apply(lambda row: np.concatenate((row["Embeddings_Trama"], row["Embeddings_Genere"], row["Embeddings_Regia"], row["Embeddings_Attori"], row["Embeddings_Tipologia"])), axis=1)
        embeddings["allEmbeddings"] = embeddings["allEmbeddings"].apply(lambda x: StandardScaler().fit_transform(x.reshape(-1, 1)).reshape(-1))
        embeddings["allEmbeddings"] = reducePCA(embeddings, "allEmbeddings", 1024)

        embeddings = embeddings.drop(columns=["Embeddings_Genere", "Embeddings_Regia", "Embeddings_Attori", "Embeddings_Tipologia"])
        embeddings.to_parquet("embeddings.parquet")
    #* --------------------------------------------


    #* ---------- Naive Raccomender based on similarity --------
    if args.qualitative:
        while((title := input("Inserisci il titolo del film: ").strip()) not in movies.Titolo.values):
            print("Film non trovato")

        # title = "The Conjuring - Il caso Enfield"

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
    remaining = [elem for elem in utilityMatrix.columns if elem not in columns]

    validUsers = None
    for col in columns:
        validUsers = utilityMatrix[utilityMatrix[col] != 0] if validUsers is None else validUsers[validUsers[col] != 0]

    train = validUsers[columns]
    test = validUsers[remaining]
    #* ---------------------------------------


    #* ---------- Prediction -----------------
    if args.algorithm in ["linear", "knn", "ordinal"]:
        for bias in [False, True]:
            rmse, mae = predict(train, test, embeddings, model=args.algorithm, kneighbors=args.count, bias=bias, round=args.round)

            if 0 in [rmse.size, mae.size]:
                print("No ratings found")
                return

            print()
            print(f"{args.algorithm.capitalize()} Regression {'with' if bias else 'without'} bias: ")
            print(f"MAE: {mae.mean()}")
            print(f"RMSE: {rmse.mean()}")
    elif args.algorithm != "":
        raise Exception("Model not supported")
    #* ----------------------------------------


    #*----------- Pure Theoric Approach -------------
    if args.user:
        def cossim(X, Y):
            return np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y))

        userProfile, res = predictWithUser(train, embeddings)

        test = test.sample(frac=1, random_state=42)
        test = test.iloc[0].T
        dfTest = pd.DataFrame({
            "Titolo" : test.index,
            "allEmbeddings" : [embeddings[embeddings.Titolo == elem].allEmbeddings.values[0] for elem in test.index],
            "Rating" : test.values,
        })
        films = dfTest[dfTest.Rating != 0].allEmbeddings.values

        totSim = np.empty(0)
        for film in films:
            totSim = np.append(totSim, cossim(userProfile, film))

        print("L'utente ha visto questi film:")
        print(res)
        print()
        print(movies.iloc[np.argsort(totSim)[-11::]]["Titolo"][-2::-1])
    #* ----------------------------------------------



if __name__ == "__main__":
    main()
