import numpy as np
import pandas as pd

import mord as md
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor as KNNRegressor





def predict(train, test, embeddings, **kwargs) -> (np.array, np.array):
    if kwargs["model"] == "linear":
        model = LinearRegression(n_jobs=-1)
    elif kwargs["model"] == "knn": 
        model = KNNRegressor(n_jobs=-1, n_neighbors=kwargs["kneighbors"], metric="cosine", weights="distance")
    elif kwargs["model"] == "ordinal":
        model = md.OrdinalRidge()
    else:
        raise Exception("Model not supported")

    preds = np.array([])
    ratings = np.array([])
    for ((_, rowX), (_, rowY)) in zip(train.iterrows(), test.iterrows()):
        dfTrain = pd.DataFrame({
            "Titolo" : train.columns,
            "allEmbeddings" : [embeddings[embeddings.Titolo == elem].allEmbeddings.values[0] for elem in train.columns],
            "Rating" : rowX.values,
        })
        dfTest = pd.DataFrame({
            "Titolo" : test.columns,
            "allEmbeddings" : [embeddings[embeddings.Titolo == elem].allEmbeddings.values[0] for elem in test.columns],
            "Rating" : rowY.values,
        })
        dfTest = dfTest[dfTest.Rating != 0]

        if kwargs["model"] == "ordinal":
            catType = pd.api.types.CategoricalDtype(categories=[1, 2, 3, 4, 5], ordered=True)
            dfTrain.Rating = dfTrain.Rating.astype(catType)
            dfTest.Rating = dfTest.Rating.astype(catType)

        model.fit(dfTrain.allEmbeddings.tolist(), dfTrain.Rating.tolist())
        pred = model.predict(dfTest.allEmbeddings.tolist())
        pred = np.rint(pred)

        preds = np.append(preds, pred)
        ratings = np.append(ratings, dfTest.Rating.values)

    return preds, ratings
