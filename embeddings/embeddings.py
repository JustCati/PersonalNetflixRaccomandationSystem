import time
import openai
import numpy as np

from sentence_transformers import SentenceTransformer


def getEmbeddings_E5_LargeV2(df):
    model = SentenceTransformer('embaas/sentence-transformers-e5-large-v2')
    for index, row in df.iterrows():
        if (row.Embedding == np.inf).all(): 
            try:
                embedding = model.encode(row.Trama)
                df.at[index, "Embedding"] = embedding
                df.to_parquet("dataset.parquet")
                print(f"Embedding {index} done!")
            except Exception as e:
                print("Error:", e)
                print("Saving Dataset...")
                df.to_parquet("dataset.parquet")
                exit()


def getEmbeddingsOpenAI(df, API_KEY):
    openai.api_key = API_KEY
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
