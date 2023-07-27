import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances



def getSimilarityMatrix(df, colName="Embeddings_Trama", method="cosine"):
    embeddings = df[colName].values
    embeddings = np.vstack(embeddings)
    return cosine_similarity(embeddings) if method == "cosine" else euclidean_distances(embeddings)

