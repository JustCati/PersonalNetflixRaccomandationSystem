import os
import numpy as np
import pandas as pd

from distances import *
from embeddings import *
from dataset.dataParser import getDataset

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
        embeddings = pd.DataFrame(movies.id, columns=["id"])
        embeddings["Embeddings_Trama"] = [np.full((1024,), np.inf, dtype=np.float32) for _ in range(len(movies))]



    #*---------- Get Embedding for Trama ------------
    toUpdate = False
    if (embeddings.Embeddings_Trama.values[0] == [np.full((1024,), np.inf, dtype=np.float32) for _ in range(len(movies))]).all():
        embeddings = getEmbeddingsTrama_E5_LargeV2(embeddings, movies, shuffle=True)
        toUpdate = True
    #*-----------------------------------------------

    #*---------- Get Vector for Genere, Regia, Attori, Tipologia -----------
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
    #*--------------------------------------------
    
        
    #*---------- Naive --------
    while((title := input("Inserisci il titolo del film: ").strip()) not in movies.Titolo.values):
        print("Film non trovato")
    id = movies[movies.Titolo == title].id.values[0]
    index = movies[movies.Titolo == title].index[0]
    
    
    for distance in ("cosine", "euclidean"):
        similarity_matrix = getSimilarityMatrix(embeddings, colName="allEmbeddings", method=distance)
        
        print(f"Film simili con {distance} come distanza:")
        if distance == "cosine":
            print(movies.iloc[np.argsort(similarity_matrix[index])[-11::]]["Titolo"][-2::-1])
        elif distance == "euclidean":
            print(movies.iloc[np.argsort(similarity_matrix[index])[10::-1]]["Titolo"][-2::-1])

    #! Dot Product (unused because embeddings is scaled so dot product is the same as cosine similarity)
    movie = embeddings.allEmbeddings.values[index]
    toCalculate = np.array([np.array(elem) for elem in embeddings.allEmbeddings.values])
    
    similar = toCalculate.dot(movie)
    sorted_idx = np.argsort(-similar)
    
    print(f"Film simili con dot product:")
    print(movies.iloc[sorted_idx[10::-1]]["Titolo"][-2::-1])
    #*-------------------------------
    


if __name__ == "__main__":
    main()
