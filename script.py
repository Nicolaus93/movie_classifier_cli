import argparse
import joblib
from typing import Mapping
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier


class classifier(object):

    def __init__(self, data_path: Path, max_features: int=1000, test_size: float=.3, verbose: bool=True):
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

    def load(self):
        """Load the model and vectorizer previously stored.
        """
        if self.verbose:
            print("Loading model..")
        self.model = joblib.load(self.data_path / "model.pkl")
        self.vect = joblib.load(self.data_path / "vect.pkl")
        return

    def train(self, features, targets, genres, save: bool=True):
        """Train a Random Forest Classifier.
        """
        # Todo: adding checks for features and targets

        # initialize the random forest and vectorizer
        self.model = RandomForestClassifier()
        self.vectorizer = TfidfVectorizer(max_features = 1000, stop_words='english', lowercase=True)

        # split dataset
        X_train, X_valid, y_train, y_valid = train_test_split(
            features, targets, test_size=self.test_size, random_state=42)

        # transform descriptions into arrays
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_valid_vec = self.vectorizer.transform(X_valid)

        # fit the model
        self.model.fit(X_train_vec, y_train)

        # display results and training and validation set
        if self.verbose:
            train_pred = self.model.predict(X_train_vec)
            valid_pred = self.model.predict(X_valid_vec)

            # print results
            print("Classification Report")
            print("Training:\n", classification_report(y_true=y_train, y_pred=train_pred, target_names=genres))
            print("Validation:\n", classification_report(y_true=y_valid, y_pred=valid_pred, target_names=genres))
            print("Accuracy")
            train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
            valid_acc = accuracy_score(y_true=y_valid, y_pred=valid_pred)
            print("Traning: ", train_acc)
            print("Validation: ",valid_acc)

        # save
        if save:
            joblib.dump(self.model, self.data_path / "model.pkl")
            joblib.dump(self.vect, self.data_path / "vect.pkl")
        return

    def predict(self, title: str, description: str) -> Mapping[str, str]:
        """Predict movie genre based on description.
        """
        assert type(title) == type(description) == str, "Please provide title and description as strings"
        if len(title) == 0:
            raise ValueError("Please provide a title!")
        if len(description) == 0:
            raise ValueError("Please provide a description!")
        # todo: raise Error is model or vectorizer = None
        if self.verbose:
            print("We're working for you...")
        feat_vec = self.vect.transform([description])
        genre = self.model.predict(feat_vec)
        result = {"title": title, "description": description, "genre": genre}
        return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--title', default='', type=str, help='The title of the movie')
    parser.add_argument('--description', default='', type=str, help='The description of the movie')
    args = parser.parse_args()
    title = args.title
    description = args.description

    p = Path('.')
    simple_model = classifier(p)
    simple_model.load()
    res = simple_model.predict(title, description)
    print(res)
    # The evil Iago pretends to be friend of Othello in order to manipulate him to serve his own end in the film version of this Shakespeare classic.