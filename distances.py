import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def getSimilarityMatrix(df, colName="Embeddings_Trama"):
    embeddings = df[colName].values
    embeddings = np.vstack(embeddings)
    return cosine_similarity(embeddings)
