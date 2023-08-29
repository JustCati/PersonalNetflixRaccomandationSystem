import os
import random
import argparse
import numpy as np
import pandas as pd

from raccomend.distances import *
from raccomend.embeddings import *
from raccomend.predict import predict

from dataset.dataScraper import getDataset
from dataset.raccomenderDataset import getUtilityMatrix

from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, ndcg_score, mean_absolute_error





def main():
    parser = argparse.ArgumentParser(description="Raccomender")
    parser.add_argument("-q", "--qualitative", action="store_true", default=None, help="Show naive qualitative raccomendation")
    parser.add_argument("-u", "--user", action="store_true", default=None, help="Show results with user profile similarity")
    parser.add_argument("-c", "--count", type=int, default=1, help="Set the number of ratings to use for training for each user")
    parser.add_argument("-a", "--algorithm", type=str, default="", help="Set the algorithm to use for raccomendation")
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

        for col in ["Embeddings_Genere", "Embeddings_Regia", "Embeddings_Attori", "Embeddings_Tipologia"]:
            embeddings[col] = reduceCategorial(embeddings, col, 1024)
        embeddings.to_parquet("embeddings.parquet")

    embeddings["allEmbeddings"] = embeddings.apply(lambda row: np.concatenate((row["Embeddings_Trama"], row["Embeddings_Genere"], row["Embeddings_Regia"], row["Embeddings_Attori"], row["Embeddings_Tipologia"])), axis=1)
    embeddings["allEmbeddings"] = reducePCA(embeddings, "allEmbeddings", 1024)
    #* --------------------------------------------


    #* ---------- Naive Raccomender based on similarity --------
    if args.qualitative is not None:
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
    remaining = [elem for elem in utilityMatrix.columns.tolist() if elem not in columns]

    validUsers = None
    for col in columns:
        validUsers = utilityMatrix[utilityMatrix[col] != 0] if validUsers is None else validUsers[validUsers[col] != 0]

    train = validUsers[columns]
    test = validUsers[remaining]
    #* ---------------------------------------


    #* ---------- Prediction -------------
    if args.algorithm in ["linear", "knn", "ordinal"]:
        preds, ratings = predict(train, test, embeddings, model=args.algorithm, kneighbors=args.count)

        if preds.shape[0] == 0:
            print("No ratings found")
            return

        print()
        print(f"{args.algorithm.capitalize()} Regression: ")
        print(f"MAE: {mean_absolute_error(ratings, preds)}")
        print(f"RMSE: {mean_squared_error(ratings, preds, squared=False)}")
        print(f"NDCG: {ndcg_score([ratings], [preds])}")
        print(f"Spearman Correlation: {spearmanr(preds, ratings).statistic}")
        print(f"Ratings predicted: {set(np.rint(preds))}")
    #* ----------------------------------------


    #*-----------------------------------------
    if args.user is not None:
        for ((_, rowX), (_, rowY)) in zip(train.iterrows(), test.iterrows()):
            print(f"User: {rowX.name}")
            print(f"Film visti: {rowX[rowX != 0].index.values}")
            print(f"Ratings: {rowX[rowX != 0].values}")


            dfTrain = pd.DataFrame({
                "Titolo" : train.columns,
                "allEmbeddings" : [embeddings[embeddings.Titolo == elem].allEmbeddings.values[0] for elem in train.columns],
                "Rating" : rowX.values,
            })
            dfTest = pd.DataFrame({
                "Titolo" : test.columns,
                "allEmbeddings" : [embeddings[embeddings.Titolo == elem].allEmbeddings.values[0] for elem in test.columns],
                "Rating" : rowY.values,
            })
            dfTest = dfTest[dfTest.Rating != 0]

            meanRating = dfTrain.Rating.mean()
            normalizedRating = dfTrain.Rating.values - meanRating

            userProfile = np.empty((len(normalizedRating), 1024))
            for i, elem, embedding in zip(range(args.count), normalizedRating, dfTrain.allEmbeddings.values):
                userProfile[i] = np.array([elem]) * embedding
            userProfile = userProfile.mean(axis=0)

            for elem, embedding in zip(dfTest.Titolo.values, dfTest.allEmbeddings.values):
                cossim = cosine_similarity([userProfile], [embedding])[0][0]

                print(f"Film: {elem}")
                print(f"Rating True: {dfTest[dfTest.Titolo == elem].Rating.values[0]}")
                print(f"Similarità: {cossim}")
                print()

            exit()
    #* ---------------------------------------



if __name__ == "__main__":
    main()
