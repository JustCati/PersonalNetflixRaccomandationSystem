import numpy as np
import pandas as pd

import mord as md
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor as KNNRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error



def predict(train, test, embeddings, **kwargs):
    if "model" in kwargs:
        if kwargs["model"] == "linear":
            model = LinearRegression(n_jobs=-1)
        elif kwargs["model"] == "knn": 
            model = KNNRegressor(n_jobs=-1, n_neighbors=kwargs["kneighbors"], metric="cosine", weights="distance")
        elif kwargs["model"] == "ordinal":
            model = md.OrdinalRidge()
        else:
            raise Exception("Model not supported")
    else:
        raise Exception("Model not specified")

    rmses = np.empty(0)
    maes = np.empty(0)
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

        model.fit(dfTrain.allEmbeddings.to_list(), dfTrain.Rating.to_list())
        pred = model.predict(dfTest.allEmbeddings.to_list())
        pred = np.clip(pred, 1, 5)

        if "round" in kwargs and kwargs["round"] == True:
            pred = np.round(pred)

        if "bias" in kwargs and kwargs["bias"] == True:
            dfTest = dfTest.reset_index(drop=True)

            if kwargs["model"] == "ordinal":
                dfTest.Rating = dfTest.Rating.astype(int)

            exceeded = dfTest.Rating < pred
            defect = dfTest.Rating > pred

            exceeded = exceeded[exceeded == True]
            defect = defect[defect == True]
            
            exceeded = dfTest.iloc[exceeded.index].Rating - pred[exceeded.index]
            defect = dfTest.iloc[defect.index].Rating - pred[defect.index]
                
            bias = np.nanmean(np.concatenate((exceeded, defect)))
            pred = pred + bias

        rmse = mean_squared_error(dfTest.Rating, pred, squared=False)
        mae = mean_absolute_error(dfTest.Rating, pred)

        rmses = np.append(rmses, rmse)
        maes = np.append(maes, mae)

    return rmses, maes


def predictWithUser(train, embeddings):
    train = train.sample(frac=1, random_state=42)
    train = train.iloc[0].T

    dfTrain = pd.DataFrame({
        "Titolo" : train.index,
        "allEmbeddings" : [embeddings[embeddings.Titolo == elem].allEmbeddings.values[0] for elem in train.index],
        "Rating" : train.values,
    })

    meanRating = dfTrain.Rating.mean()
    normalizedRating = dfTrain.Rating.values - meanRating

    userProfile = np.empty((len(normalizedRating), 1024))
    for i, elem, embedding in zip(range(len(dfTrain.allEmbeddings.values)), normalizedRating, dfTrain.allEmbeddings.values):
        userProfile[i] = np.array([elem]) * embedding
    userProfile = userProfile.mean(axis=0)

    return userProfile, {dfTrain.Titolo[i] : dfTrain.Rating[i] for i in range(dfTrain.shape[0])}
