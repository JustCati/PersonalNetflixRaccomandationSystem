import time
import torch
import openai
import numpy as np

from sentence_transformers import SentenceTransformer


def getEmbeddings_E5_LargeV2(df, shuffle=True):
    if torch.cuda.is_available():
        torch.device("cuda")
        print("Pytorch using CUDA")
    elif torch.backends.mps.is_available():
        torch.device("mps")
        print("Pytorch using MPS")
    else:
        torch.device("cpu")
        print("Pytorch using CPU")
    
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    model = SentenceTransformer('embaas/sentence-transformers-e5-large-v2')
    for index, row in df.iterrows():
        if (row.Embedding == np.inf).all(): 
            embedding = model.encode(row.Trama)
            df.at[index, "Embedding"] = embedding
            df.to_parquet("dataset.parquet")
            print(f"Embedding {index} done!")



def getEmbeddingsOpenAI(df, API_KEY, shuffle=True):
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
