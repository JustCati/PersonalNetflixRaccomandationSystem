import time
import openai
import numpy as np



def getEmbeddings(df, API_KEY):
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
                print(f"Embedding {index} done!")
                time.sleep(30)
            except Exception as e:
                print("Error:", e)
                print("Saving Dataset...")
                df.to_parquet("dataset.parquet")
                exit()
