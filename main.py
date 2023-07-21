import os
import numpy as np
import pandas as pd

from dataset.dataParser import getDataset
from embeddings.embeddings import getEmbeddingsOpenAI, getEmbeddings_E5_LargeV2


def main():
    if os.path.exists("dataset.parquet"):
        df = pd.read_parquet("dataset.parquet")
    else:
        df = getDataset(update=False)
        df["Embedding"] = [np.full((1024,), np.inf, dtype=np.float32) for _ in range(len(df))] #! <3
        df.to_parquet("dataset.parquet")

    #*------------------------------------- 
    getEmbeddings_E5_LargeV2(df, shuffle=True)
    # getEmbeddingsOpenAI(df, shuffle=True)
    #*-------------------------------------





if __name__ == "__main__":
    main()
