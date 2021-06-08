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

    def __init__(self, data_path: Path, max_features: int=1000, test_size: float=.3, verbose: bool=True):
        """
        TODO: Explain arguments.
        """
        if max_features < 1:
            raise ValueError("Please specify a value greater than 0 for max_features!")
        if test_size < 0 or test_size > 1:
            raise ValueError("Please specify a value greater than 0 for test_size!")
        self.data_path = data_path
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
        model_dict = joblib.load(self.data_path / "model.pkl")
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
        """
        if len(features) != len(targets):
            raise ValueError("Length of features and targets differ!")
        if len(features) < 2:
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
            title: 
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
        except IndexError:
            print("Sorry, we aren't able to classify this movie :(\n\
                Try changing the description!")
        result = {"title": title, "description": description, "genre": self.genres[genre_ind]}
        return result



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='This script is used to classify movies based on a title and description provided as inputs.')
#     parser.add_argument('--title', default='', type=str, help='The title of the movie')
#     parser.add_argument('--description', default='', type=str, help='The description of the movie')
#     args = parser.parse_args()
#     title = args.title
#     description = args.description

#     p = Path('./data')
#     simple_model = RandomForest(p)
#     simple_model.load()
#     res = simple_model.predict(title, description)
#     print(res)
#     # The evil Iago pretends to be friend of Othello in order to manipulate him to serve his own end in the film version of this Shakespeare classic.