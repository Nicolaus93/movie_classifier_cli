import argparse
import joblib
from typing import Mapping
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from numpy import where


class RandomForest(object):

    def __init__(self, data_path: str, max_features: int=1000, test_size: float=.3, verbose: bool=True):
        """
        Initialize a random forest model.
        Arguments:
            - data_path: represents the path to the folder where the data is stored.
            - max_features: represents the max number of features used in the tf-idf vectorizer
                (default to 1000)
            - test_size: represents the fraction of the training data to be used as validation set
                (default to 0.3) NOTE: at the moment this is not useful since the validation test 
                is not used.
            - verbose: whether to print information (default to True)
        """
        if max_features < 1:
            raise ValueError("Please specify a value greater than 0 for max_features!")
        if test_size < 0 or test_size > 1:
            raise ValueError("Please specify a value greater than 0 for test_size!")
        self.data_path = Path(data_path)
        self.test_size = test_size
        self.max_features = max_features
        self.model = None
        self.vect = None
        self.genres = None
        self.verbose = verbose

    def load(self) -> None:
        """Load the model and vectorizer previously stored.
        """
        if self.verbose:
            print("Loading model..")
        model_path = self.data_path / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"There is no model stored at {model_path}")
        # TODO: we should check the model is a dict 
        # containing the required keys
        model_dict = joblib.load(model_path)
        self.model = model_dict["model"]
        self.vect = model_dict["vect"]
        self.genres = model_dict["genres"]
        return

    def train(self, features, targets, genres, save: bool=True) -> None:
        """Train a Random Forest Classifier.
        Arguments:
            features: Pandas series where each value is a string
                representing the description of the corresponding movie
                which will be used for training.
            targets: Pandas DataFrame where row i is a 0/1 vector
                representing the presence of the corresponding label.
            genres: list of strings representing the genres.
            save: boolean specifying whether to save the trained model.
        """
        if len(features) != len(targets):
            raise ValueError("Length of features and targets differ!")
        if len(features) < len(genres):
            raise RuntimeError("Please consider using a bigger training set!")
        if len(genres) == 0:
            raise RuntimeError("No genres provided!") 

        # initialize the random forest and vectorizer
        self.model = RandomForestClassifier()
        self.vect = TfidfVectorizer(
            max_features=self.max_features, stop_words='english', lowercase=True)
        self.genres = genres

        # split dataset
        X_train, X_valid, y_train, y_valid = train_test_split(
            features, targets, test_size=self.test_size, random_state=42)

        # transform descriptions into arrays
        X_train_vec = self.vect.fit_transform(X_train)
        X_valid_vec = self.vect.transform(X_valid)

        # fit the model
        if self.verbose:
            print("Training the model...")
        self.model.fit(X_train_vec, y_train)

        # save the model
        if save:
            if self.verbose:
                print("Saving the model...")
            model_dict = dict()
            model_dict["model"] = self.model
            model_dict["vect"] = self.vect
            model_dict["genres"] = self.genres
            joblib.dump(model_dict, self.data_path / "model.pkl")
        if self.verbose:
            print("Done!")

        # display results on training and validation set
        if self.verbose:
            train_pred = self.model.predict(X_train_vec)
            valid_pred = self.model.predict(X_valid_vec)

            # print results
            print("Classification Report")
            print("Training:\n", classification_report(
                y_true=y_train, y_pred=train_pred, target_names=self.genres))
            print("Validation:\n", classification_report(
                y_true=y_valid, y_pred=valid_pred, target_names=self.genres))
            print("Accuracy")
            train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
            valid_acc = accuracy_score(y_true=y_valid, y_pred=valid_pred)
            print("Traning: ", train_acc)
            print("Validation: ", valid_acc)

        return

    def predict(self, title: str, description: str) -> Mapping[str, str]:
        """Predict movie genre based on description.
        Arguments:
            - title: title of the movie, it's not used for the prediction
            - description: short description of the movie, used for the prediction
        """
        if type(title) != str:
            raise TypeError("Please provide title as string.")
        if type(description) != str:
            raise TypeError("Please provide description as string.")
        if not self.model or not self.vect:
            raise RuntimeError("Model not trained!")
        if self.verbose:
            print("Now predicting...")

        # transform description into array
        feat_vec = self.vect.transform([description])

        # generate prediction
        pred = self.model.predict(feat_vec)
        try:
            genre_ind = where(pred[0] == 1)[0][0]
            result = {"title": title, "description": description, "genre": self.genres[genre_ind]}
        except IndexError:
            print("Sorry, the model was not able to classify this movie :(\n\
                Try changing the description!")
            result = {"title": title, "description": description, "genre": "Not found"}
        return result
