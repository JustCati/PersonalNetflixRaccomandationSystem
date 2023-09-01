import numpy as np
import pandas as pd

from scipy.stats import spearmanr, pearsonr

import mord as md
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor as KNNRegressor
from sklearn.metrics import mean_squared_error, ndcg_score, mean_absolute_error



def predict(train, test, embeddings, **kwargs):
    if kwargs["model"] == "linear":
        model = LinearRegression(n_jobs=-1)
    elif kwargs["model"] == "knn": 
        model = KNNRegressor(n_jobs=-1, n_neighbors=kwargs["kneighbors"], metric="cosine", weights="distance")
    elif kwargs["model"] == "ordinal":
        model = md.OrdinalRidge()
    else:
        raise Exception("Model not supported")

    rmses = np.empty(0)
    maes = np.empty(0)
    ndcgs = np.empty(0)
    spear = np.empty(0)
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
        pred = np.clip(pred, 1, 5)

        rmse = mean_squared_error(dfTest.Rating.tolist(), pred, squared=False)
        mae = mean_absolute_error(dfTest.Rating.tolist(), pred)
        ndcg = ndcg_score([dfTest.Rating.tolist()], [pred])
        spearman = spearmanr(dfTest.Rating.tolist(), pred).statistic
        
        rmses = np.append(rmses, rmse)
        maes = np.append(maes, mae)
        ndcgs = np.append(ndcgs, ndcg)
        spear = np.append(spear, spearman)

    return rmses, maes, ndcgs, spear


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

    return userProfile, {dfTrain.Titolo[i] : dfTrain.Rating[i] for i in range(len(dfTrain.Titolo))}
