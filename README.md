# Personal Content Based Recommendation System for Netflix



## Abstract


The aim of this project is to develop a recommendation system for products available on the Netflix streaming platform based on user content and preferences. The system has been created to provide a personalized recommendation alternative to Netflix's proprietary system. The goal is to achieve a recommendation system independent of advertising biases.

Additionally, the recommendation problem has been approached as a regression problem, considering scenarios where there are not many user ratings available. Two different datasets have been utilized: the first one was created through scraping to obtain metadata for movies and TV series, which were then used for data representation. The second dataset is the Netflix Prize Dataset, containing user ratings.

The main focus of the project lies on the K-NN Regressor model, which predicts user ratings for unseen movies based on where the movies are mapped in the space. The results obtained are not optimal but still acceptable, considering the model was trained on a dataset with limited data.

It is also noted that the representation of movies used is not optimal; with a more suitable representation, better results could certainly be achieved.