import os
import numpy as np
import pandas as pd

from distances import *
from embeddings import *
from dataset.dataParser import getDataset

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
    if (df.Embeddings_Trama.values[0] == [np.full((1024,), np.inf, dtype=np.float32) for _ in range(len(df))]).all():
        df = getEmbeddingsTrama_E5_LargeV2(df, shuffle=True)
    #*-----------------------------------------------

    #*---------- Get Vector for Genere, Regia, Attori, Tipologia -----------
    df = getFeatureAttori(df, colName="Attori")
    df = getFeatureTipologia(df)
    df = getFeatureGenere(df)
    df = getFeatureAttori(df, colName="Regia")
    #*-----------------------------------------------------------------------

    # The Conjuring - Il caso Enfield
    # x["Embeddings_Attori"], x["Embeddings_Tipologia"], x["Embeddings_Regia"]
    #*---------- Get Similarity Matrix ------------
    df["allEmbeddings"] = df.apply(lambda x: np.concatenate((x["Embeddings_Trama"], x["Embeddings_Genere"], x["Embeddings_Regia"], x["Embeddings_Attori"], x["Embeddings_Tipologia"]), dtype=np.float32), axis=1)
    df.allEmbeddings = df.apply(lambda x: StandardScaler().fit_transform(x["allEmbeddings"].reshape(-1, 1)).reshape(-1), axis=1)
    
    similarity_matrix = getSimilarityMatrix(df, colName="allEmbeddings")
    #*--------------------------------------------
    
    
    #*---------- Naive ------------
    while((movie := input("Inserisci il titolo del film: ").strip()) not in df.Titolo.values):
        print("Film non trovato")

    index = df[df.Titolo == movie].index[0]
    print("Film simili:")
    print(df.iloc[np.argsort(similarity_matrix[index])[-11:]]["Titolo"][-2::-1])
    #*-----------------------------


if __name__ == "__main__":
    main()
