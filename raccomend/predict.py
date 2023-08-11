import numpy as np
import pandas as pd

import mord as md
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor as KNNRegressor





def predict(train, test, embeddings, **kwargs) -> (np.array, np.array, float):
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
        dfTrain = pd.DataFrame({"Titolo" : train.columns,
                                "Embeddings_Trama" : [embeddings[embeddings.Titolo == elem].Embeddings_Trama.values[0] for elem in train.columns],
                                "Embeddings_Genere" : [embeddings[embeddings.Titolo == elem].Embeddings_Genere.values[0] for elem in train.columns],
                                "Embeddings_Regia" : [embeddings[embeddings.Titolo == elem].Embeddings_Regia.values[0] for elem in train.columns],
                                "Embeddings_Attori" : [embeddings[embeddings.Titolo == elem].Embeddings_Attori.values[0] for elem in train.columns],
                                "Embeddings_Tipologia" : [embeddings[embeddings.Titolo == elem].Embeddings_Tipologia.values[0] for elem in train.columns],
                                "Rating" : rowX.values})

        dfTest = pd.DataFrame({"Titolo" : test.columns, 
                                "Embeddings_Trama" : [embeddings[embeddings.Titolo == elem].Embeddings_Trama.values[0] for elem in test.columns],
                                "Embeddings_Genere" : [embeddings[embeddings.Titolo == elem].Embeddings_Genere.values[0] for elem in test.columns],
                                "Embeddings_Regia" : [embeddings[embeddings.Titolo == elem].Embeddings_Regia.values[0] for elem in test.columns],
                                "Embeddings_Attori" : [embeddings[embeddings.Titolo == elem].Embeddings_Attori.values[0] for elem in test.columns],
                                "Embeddings_Tipologia" : [embeddings[embeddings.Titolo == elem].Embeddings_Tipologia.values[0] for elem in test.columns],
                                "Rating" : rowY.values})
        dfTest = dfTest[dfTest.Rating != 0]

        if kwargs["model"] == "ordinal":
            catType = pd.api.types.CategoricalDtype(categories=[1, 2, 3, 4, 5], ordered=True)
            dfTrain.Rating = dfTrain.Rating.astype(catType)
            dfTest.Rating = dfTest.Rating.astype(catType)

        allEmbeddingsTrain = np.concatenate((dfTrain.Embeddings_Trama.values.tolist(),
                                            dfTrain.Embeddings_Genere.values.tolist(),
                                            dfTrain.Embeddings_Regia.values.tolist(),
                                            dfTrain.Embeddings_Attori.values.tolist(),
                                            dfTrain.Embeddings_Tipologia.values.tolist()), axis=1)
        allEmbeddingsTest = np.concatenate((dfTest.Embeddings_Trama.values.tolist(),
                                            dfTest.Embeddings_Genere.values.tolist(),
                                            dfTest.Embeddings_Regia.values.tolist(),
                                            dfTest.Embeddings_Attori.values.tolist(),
                                            dfTest.Embeddings_Tipologia.values.tolist()), axis=1)
        model.fit(allEmbeddingsTrain, dfTrain.Rating.tolist())
        pred = model.predict(allEmbeddingsTest)
        pred = np.rint(pred)

        preds = np.append(preds, pred)
        ratings = np.append(ratings, dfTest.Rating.values)

    return preds, ratings
