import os
import numpy as np
import pandas as pd

from raccomend.raccomend import *
from raccomend.embeddings import *

from dataset.dataScraper import getDataset
from dataset.raccomenderDataset import getUtilityMatrix

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main():
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
        embeddings = pd.DataFrame({"id" : movies.id, "Tipologia" : movies.Tipologia}, columns=["id", "Tipologia"])
        embeddings["Embeddings_Trama"] = [np.full((1024,), np.inf, dtype=np.float32) for _ in range(len(movies))]


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


    #* ---------- Naive --------
    while((title := input("Inserisci il titolo del film: ").strip()) not in movies.Titolo.values):
        print("Film non trovato")

    print("Film simili con cosine come distanza:")
    print(getMostSimilarCosine(movies, embeddings, title))

    print("\nFilm simili con euclidean come distanza:")
    print(getMostSimilarEuclidean(movies, embeddings, title))
    #* -------------------------------


    #* ---------- KNN-Regressor -------------
    if not os.path.exists("utilitymatrix.parquet"):
        utilityMatrix = getUtilityMatrix(movies)
        utilityMatrix.to_parquet("utilitymatrix.parquet")
    else:
        utilityMatrix = pd.read_parquet("utilitymatrix.parquet")
    #* ---------------------------------------



if __name__ == "__main__":
    main()
