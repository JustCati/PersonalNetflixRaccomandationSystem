import torch
import numpy as np
import pandas as pd

from prince import MCA, MFA

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder


def reduceCategorial(df, colName, n_components=1024):
    if len(df[colName].values[0]) > n_components:
        mca = MCA(n_components=n_components, engine="sklearn")
    else:
        mca = MCA(n_components=len(df[colName].values[0]), engine="sklearn")
    data = df[colName].values
    data = pd.DataFrame(data.tolist())
    mca.fit(data)
    data = mca.transform(data)
    return data.values.tolist()


def reduceMFA(df, colName, n_components=1024):
    mfa = MFA(n_components=n_components, engine="sklearn")
    data = df[colName].values
    data = pd.DataFrame(data.tolist())
    groups = {
        "allEmbeddings" : [index for index in range(len(data.columns))]
    }
    mfa.fit(data, groups=groups)
    data = mfa.transform(data)
    return data.values.tolist()


def encode(trama, model):
    if len(trama.split(" ")) > 512:
        totalEmbeddings = []
        for line in trama.split("\n"):
            totalEmbeddings.append(model.encode("query: " + line))
        return np.array(totalEmbeddings).mean(axis=0)
    else:
        return model.encode("query: " + trama)


def getEmbeddingsTrama_E5_LargeV2(df, movies, shuffle=True):
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    model = SentenceTransformer('intfloat/multilingual-e5-large', device=("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"))
    df.Embeddings_Trama = df.apply((lambda x: encode(movies[(movies.id == x.id) & (movies.Tipologia == x.Tipologia)].Trama.values[0], model) if (x.Embeddings_Trama == np.inf).all() else x.Embeddings_Trama), axis=1)
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
