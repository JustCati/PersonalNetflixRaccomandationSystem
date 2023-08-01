import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor as KNNRegressor



def predict(train, test, embeddings, remaining, **kwargs) -> (np.array, np.array):
    if kwargs["model"] == "linear":
        model = LinearRegression(n_jobs=-1)
    elif kwargs["model"] == "knn": 
        model = KNNRegressor(n_jobs=-1, n_neighbors=kwargs["kneighbors"], metric="cosine")
    else:
        raise Exception("Model not supported")

    preds = np.array([])
    ratings = np.array([])
    
    for ((_, rowX), (_, rowY)) in zip(train.iterrows(), test.iterrows()):
        dfTrain = pd.DataFrame({"Titolo" : train.columns, "Embeddings" : [embeddings[embeddings.Titolo == elem].allEmbeddings.values[0] for elem in train.columns], "Rating" : rowX.values})
        dfTest = pd.DataFrame({"Titolo" : test.columns, "Embeddings" : [embeddings[embeddings.Titolo == rem].allEmbeddings.values[0] for rem in remaining], "Rating" : rowY.values})
        dfTest = dfTest[dfTest.Rating != 0]

        model.fit(dfTrain.Embeddings.tolist(), dfTrain.Rating.tolist())
        pred = model.predict(dfTest.Embeddings.tolist())
        
        preds = np.append(preds, pred)
        ratings = np.append(ratings, dfTest.Rating.values)
        
    return preds, ratings
