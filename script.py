import argparse
import joblib
from typing import Mapping
from pathlib import Path


class classifier(object):

    def __init__(self, data_path: Path, verbose: bool=True):
        self.data_path = data_path
        self.model = None
        self.vect = None
        self.verbose = verbose

    def load(self):
        """Load the model and vectorized previously stored.
        """
        if self.verbose:
            print("Loading model..")
        self.model = joblib.load(self.data_path / "model.pkl")
        self.vect = joblib.load(self.data_path / "vect.pkl")

    def train(self):
        return

    def predict(self, title: str, description: str) -> Mapping[str, str]:
        """Predict movie genre based on description.
        """
        assert type(title) == type(description) == str, "Please provide title and description as strings"
        if len(title) == 0:
            raise ValueError("Please provide a title!")
        if len(description) == 0:
            raise ValueError("Please provide a description!")
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