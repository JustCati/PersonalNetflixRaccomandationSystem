import os
import sys
import numpy as np
import pandas as pd

from dataset.dataParser import getDataset
from embeddings.embeddings import getEmbeddings


def main():
    if os.path.exists("dataset.parquet"):
        df = pd.read_parquet("dataset.parquet")
    else:
        df = getDataset()
        df["Embedding"] = [np.full((1536,), np.inf) for _ in range(len(df))] #! <3
        df.to_parquet("dataset.parquet")
    
    #*-------------------------------------

    API_KEY = sys.argv[1]
    if ".env" in API_KEY:
        with open(API_KEY) as f:
            API_KEY = f.read().strip()
    API_KEY = API_KEY.split("=")[1]

    getEmbeddings(df, API_KEY)







if __name__ == "__main__":
    main()
