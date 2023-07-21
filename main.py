import os
import numpy as np
import pandas as pd

from embeddings.embeddings import *
from dataset.dataParser import getDataset


def main():
    if os.path.exists("netflix.parquet"):
        df = pd.read_parquet("netflix.parquet")
    elif os.path.exists("netflixOriginal.parquet"):
        original_df = pd.read_parquet("netflixOriginal.parquet")
        df = original_df.copy(deep=True)
        df["Embeddings_Trama"] = [np.full((1024,), np.inf, dtype=np.float32) for _ in range(len(df))]
    else:
        original_df = getDataset(update=False) #! DEBUG, change with update=True
        original_df.to_parquet("netflix.parquet")
        df["Embeddings_Trama"] = [np.full((1024,), np.inf, dtype=np.float32) for _ in range(len(df))]


    #*---------- Get Embedding for Trama ------------
    df = getEmbeddingsTrama_E5_LargeV2(df, shuffle=True)
    #*-----------------------------------------------



if __name__ == "__main__":
    main()
