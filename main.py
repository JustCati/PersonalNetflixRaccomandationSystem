import os
import numpy as np
import pandas as pd

from distances import *
from embeddings import *
from dataset.dataParser import getDataset

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main():
    if os.path.exists("netflix.parquet"):
        df = pd.read_parquet("netflix.parquet")
    elif os.path.exists("netflixOriginal.parquet"):
        original_df = pd.read_parquet("netflixOriginal.parquet")
        df = original_df.copy(deep=True)
        df["Embeddings_Trama"] = [np.full((1024,), np.inf, dtype=np.float32) for _ in range(len(df))]
    else:
        original_df = getDataset(update=False)              #! DEBUG, change with update=True
        original_df.to_parquet("netflix.parquet")
        df["Embeddings_Trama"] = [np.full((1024,), np.inf, dtype=np.float32) for _ in range(len(df))]



        #*---------- Get Embedding for Trama ------------
    toUpdate = False
    if (df.Embeddings_Trama.values[0] == [np.full((1024,), np.inf, dtype=np.float32) for _ in range(len(df))]).all():
        df = getEmbeddingsTrama_E5_LargeV2(df, shuffle=True)
        toUpdate = True
    #*-----------------------------------------------

    #*---------- Get Vector for Genere, Regia, Attori, Tipologia -----------
    if toUpdate or "allEmbeddings" not in df.columns:
        df = getFeatureGenere(df)
        df = getFeatureAttori(df, colName="Regia")
        df = getFeatureAttori(df, colName="Attori")
        df = getFeatureTipologia(df)

        df["allEmbeddings"] = df.apply(lambda x: np.concatenate((x["Embeddings_Trama"], x["Embeddings_Genere"], x["Embeddings_Regia"], x["Embeddings_Attori"], x["Embeddings_Tipologia"]), dtype=np.float32), axis=1)
        df = df.drop(columns=["Embeddings_Trama", "Embeddings_Genere", "Embeddings_Regia", "Embeddings_Attori", "Embeddings_Tipologia"])
        df.allEmbeddings = df.apply(lambda x: StandardScaler().fit_transform(x["allEmbeddings"].reshape(-1, 1)).reshape(-1), axis=1)

        pca = PCA(n_components=1024)
        data = df.allEmbeddings.values
        data = np.array([np.array(elem) for elem in data])

        pca.fit(data)
        data = pca.transform(data)
        df.allEmbeddings = data.tolist()
    if "Embeddings_Trama" in df.columns:
        df = df.drop(columns=["Embeddings_Trama"]) 
    #*--------------------------------------------
    
        
    #*---------- Naive --------
    while((title := input("Inserisci il titolo del film: ").strip()) not in df.Titolo.values):
            print("Film non trovato")
    index = df[df.Titolo == title].index[0]
    
    for distance in ("cosine", "euclidean"):
        similarity_matrix = getSimilarityMatrix(df, colName="allEmbeddings", method=distance)
        
        print(f"Film simili con {distance} come distanza:")
        if distance == "cosine":
            print(df.iloc[np.argsort(similarity_matrix[index])[-11::]]["Titolo"][-2::-1])
        elif distance == "euclidean":
            print(df.iloc[np.argsort(similarity_matrix[index])[10::-1]]["Titolo"][-2::-1])

    #! Dot Product (unused because embeddings is scaled so dot product is the same as cosine similarity)
    movie = df.allEmbeddings.values[index]
    embeddings = np.array([np.array(elem) for elem in df.allEmbeddings.values])
    
    similar = embeddings.dot(movie)
    sorted_idx = np.argsort(-similar)
    
    print(f"Film simili con dot product:")
    print(df.iloc[sorted_idx[10::-1]]["Titolo"][-2::-1])
    #*-------------------------------
    


if __name__ == "__main__":
    main()
