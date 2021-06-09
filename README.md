# Movie classifier

This repository contains a package implementing a command line interface used to classify movies given a title and a description in input. To install the package clone the repository and simply run `python setup.py install` or `pip install .` inside the directory.

## Usage

**NOTE**: the package will use data downloaded from *Kaggle*, hence the user needs to be registered on *Kaggle* and accept their terms and conditions. In order to access the movies data, the *Kaggle* API is used. For this to work, an API Token should be present on the system. For more details on how to do this, please refer to the official [Kaggle api](https://github.com/Kaggle/kaggle-api#api-credentials).

Once installed, you can use the package by invoking the command `movie_classifier` from the command line and providing `--title` and `--description` arguments, i.e.
```
movie_classifier --title 'Othello' --description 'The evil Iago pretends to be friend of Othello in order to manipulate him to serve his own end in the film version of this Shakespeare classic.'
```
The first time running this command, the package will download the data about movies from *Kaggle*, train the a random forest model (displaying all the statistics about the training process) and save it in the user home directory in a folder named `movie_classifier_data`.

Later, any other time the command is invoked the model will be loaded from `movie_classifier_data` and used directly for prediction.

## Libraries/Algorithms used

- `pandas` and `numpy` for preprocessing data
- `joblib` to save and load the model
- `scikit_learn`: `RandomForestClassifier` for the implementation of the classification algorithm, `TfidfVectorizer` to generate numerical features starting from the description text of the movie. Parameters of the classifier have not been tuned and default values are used. When generating numerical features with the vectorizer, we remove english stopwords (which are not relevant for the model) and fix the size of the feature vectors to `n=1000`. Note that the genre classification problem can be seen as a type of multilabel prediction problem, since a movie can be labeled with multiple genres. However, for simplicity we treat it as a multiclass classification problem, labeling the movie with the highest scoring genre.

## Complete Demonstration

After cloning the repository and making sure a *Kaggle* API Token is present on the system, run `sh example.sh`.
