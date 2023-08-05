import torch
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder


def getEmbeddingsTrama_E5_LargeV2(df, movies, shuffle=True):
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    model = SentenceTransformer('airnicco8/xlm-roberta-en-it-de', device=("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"))
    df.Embeddings_Trama = df.apply((lambda x: model.encode(movies[(movies.id == x.id) & (movies.Tipologia == x.Tipologia)].Trama.values[0]) if (x.Embeddings_Trama == np.inf).all() else x.Embeddings_Trama), axis=1)
    df.to_parquet("embeddings.parquet")
    return df


def getFeatureAttori(df, movies, colName="Attori"):
    movies[colName + "Temp"] = movies[colName].fillna("").apply(lambda x: x.split(", "))

    encoder = MultiLabelBinarizer()
    encoder.fit(movies[colName + "Temp"])
    df["Embeddings_" + colName] = encoder.transform(movies[colName + "Temp"]).tolist()
    movies = movies.drop(columns=[colName + "Temp"])
    return df


def getFeatureTipologia(df, movies):
    encoder = OneHotEncoder(handle_unknown='ignore')

    encoder.fit(movies.Tipologia.values.reshape(-1, 1))
    df["Embeddings_Tipologia"] = encoder.transform(movies.Tipologia.values.reshape(-1, 1)).toarray().tolist()
    return df


def getFeatureGenere(df, movies):
    movies["GenereTemp"] = movies.Genere.fillna("").apply(lambda x: x.replace(",", " ").split(" "))
    movies.GenereTemp = movies.GenereTemp.apply(lambda x: [i.strip() for i in x if i != ""])

    encoder = MultiLabelBinarizer()
    encoder.fit(movies.GenereTemp)
    df["Embeddings_Genere"] = encoder.transform(movies.GenereTemp).tolist()
    movies = movies.drop(columns=["GenereTemp"])
    return df 
