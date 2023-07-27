import numpy as np
from .distances import getSimilarityMatrix


def getMostSimilarCosine(movies, embeddings, title):
    index = movies[movies.Titolo == title].index[0]
    similarity_matrix = getSimilarityMatrix(embeddings, colName="allEmbeddings", method="cosine")
    return movies.iloc[np.argsort(similarity_matrix[index])[-11::]]["Titolo"][-2::-1]
    
        
        
def getMostSimilarEuclidean(movies, embeddings, title):
    index = movies[movies.Titolo == title].index[0]
    similarity_matrix = getSimilarityMatrix(embeddings, colName="allEmbeddings", method="euclidean")
    return movies.iloc[np.argsort(similarity_matrix[index])[10::-1]]["Titolo"][-2::-1]
