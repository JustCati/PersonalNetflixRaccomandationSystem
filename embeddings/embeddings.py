import sys
import time
import torch
import openai
import numpy as np

from sentence_transformers import SentenceTransformer


def getEmbeddings_E5_LargeV2(df, shuffle=True):
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    model = SentenceTransformer('embaas/sentence-transformers-e5-large-v2', device=("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"))
    for index, row in df.iterrows():
        if (row.Embedding == np.inf).all(): 
            embedding = model.encode(row.Trama)
            df.at[index, "Embedding"] = embedding
            df.to_parquet("dataset.parquet")
            print(f"Embedding {index} done!")



def getEmbeddingsOpenAI(df, shuffle=True):
    API_KEY = None
    if len(sys.argv) > 1:
        API_KEY = sys.argv[1]
        if ".env" in API_KEY:
            with open(API_KEY) as f:
                API_KEY = f.read().strip()
        API_KEY = API_KEY.split("=")[1]
    openai.api_key = API_KEY

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    for index, row in df.iterrows():
        if (row.Embedding == np.inf).all(): 
            try:
                response = openai.Embedding.create(
                    model= "text-embedding-ada-002",
                    input=[row.Trama]
                )
                embedding = response["data"][0]["embedding"]
                df.at[index, "Embedding"] = embedding
                df.to_parquet("dataset.parquet")
                print(f"Embedding {index} done!")
                time.sleep(30)
            except Exception as e:
                print("Error:", e)
                print("Saving Dataset...")
                df.to_parquet("dataset.parquet")
                exit()
