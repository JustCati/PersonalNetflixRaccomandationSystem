import os
import sys
import numpy as np
import pandas as pd

from dataset.dataParser import getDataset
from embeddings.embeddings import getEmbeddingsOpenAI, getEmbeddings_E5_LargeV2


def main():
    if os.path.exists("dataset.parquet"):
        df = pd.read_parquet("dataset.parquet")
    else:
        df = getDataset()
        df["Embedding"] = [np.full((1024,), np.inf) for _ in range(len(df))] #! <3
        df.to_parquet("dataset.parquet")

    #*-------------------------------------
    API_KEY = None
    if len(sys.argv) > 1:
        API_KEY = sys.argv[1]
        if ".env" in API_KEY:
            with open(API_KEY) as f:
                API_KEY = f.read().strip()
        API_KEY = API_KEY.split("=")[1]

    
    getEmbeddings_E5_LargeV2(df, shuffle=True)
    # getEmbeddingsOpenAI(df, API_KEY, shuffle=True)
    #*-------------------------------------





if __name__ == "__main__":
    main()
