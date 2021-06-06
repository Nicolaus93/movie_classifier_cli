import argparse
from pathlib import Path
from random_forest import RandomForest
from dataset import KaggleDataset


class CLInterface(object):

    def __init__(self, data_path: Path):
        """ 
        Initialize an interface.
        Arguments:
            data_path: Path specifying where the data is stored.
        """
        self.parser = argparse.ArgumentParser(
            description='This script is used to classify movies based on a title and description provided as inputs.')
        self.parser.add_argument(
            '--title', default='', type=str, help='The title of the movie')
        self.parser.add_argument(
            '--description', default='', type=str, help='The description of the movie')
        self.model = None
        self.title = None
        self.description = None
        self.data_path = data_path

    def parse(self):
        """Parse title and description from command line.
        """
        args = self.parser.parse_args()
        self.title = args.title
        self.description = args.description
        if len(self.title) == 0:
            raise ValueError("Please provide a title!")
        if len(self.description) == 0:
            raise ValueError("Please provide a description!")
        
    def predict(self):
        """
        Provide a prediction using the movie description.
        The function loads the model if already present,
        otherwise trains a new model (a random forest classifier).
        """
        if not self.title or not self.description:
            raise ValueError("Title and/or description not specified!")
        # retrieve model if trained, train it otherwise
        model_path = self.data_path / "model.pkl"
        self.model = RandomForest(self.data_path)
        if not model_path.exists():
            print("Model not trained yet. Training now, it can take a while..")
            dataset = KaggleDataset(self.data_path)  # create dataset
            dataset.download_and_extract()           # download and extract data
            X, Y, genres = dataset.preprocess()      # preprocess data
            self.model.train(X, Y, genres)           # train classifier
        else:
            self.model.load()  # load model
        return self.model.predict(self.title, self.description)


if __name__ == "__main__":
    p = Path('.') / "data"
    new_cli = CLInterface(p)
    new_cli.parse()
    res = new_cli.predict()
    print(res)
