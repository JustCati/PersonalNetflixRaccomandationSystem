import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances




def getSimilarityMatrix(df, colName="Embeddings_Trama", method="cosine"):
    embeddings = df[colName].values
    embeddings = np.vstack(embeddings)
    return cosine_similarity(embeddings) if method == "cosine" else euclidean_distances(embeddings)


# def getSimilaritiesSVM(df, colName="Embeddings_Trama"):
#     x = np.vstack(df[colName].values)
#     y = np.zeros(len(df))
#     y[0] = 1
    
#     clf = LinearSVC(class_weight='balanced', verbose=False, max_iter=100000, tol=1e-6, C=0.1, dual="auto")
#     clf.fit(x, y)
    
#     similarities = clf.decision_function(x)
#     return similarities
