import kaggle
import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer


class KaggleDataset(object):

    def __init__(self, storing_path):
        self.data_path = storing_path
    
    def download_and_extract(self):
        """Download the movies dataset https://www.kaggle.com/rounakbanik/the-movies-dataset from Kaggle.
        """
        # create data directory
        self.data_path.mkdir(parents=True, exist_ok=True)
        movies_metadata = self.data_path / "movies_metadata.csv"

        # download data if not present
        if not movies_metadata.exists():
            print("Downloading data..")
            # use kaggle api to download data
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files('rounakbanik/the-movies-dataset', path=self.data_path, unzip=True)

            # remove unnecessary files
            unused_files = [i for i in self.data_path.iterdir() if i != self.data_path / "movies_metadata.csv"]
            for f in unused_files:
                f.unlink()

        # update data path
        self.data_path = self.data_path / "movies_metadata.csv"

    def preprocess(self):
        """
        Function to preprocess the movies dataset.
        Largely inspired to https://www.kaggle.com/neerajkhadagade/predicting-the-movie-genres.
        """
        print("Preprocessing data..")
        # read the csv
        df = pd.read_csv(self.data_path)

        # drop unnecessary columns
        df = df[['title', 'overview', 'genres']]
        df.set_index('title',inplace = True)
        df.dropna(subset=['overview'], inplace=True)
        df['genres'] = df['genres'].apply(literal_eval).apply(
            lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        
        # Selecting only those rows which have an actual genre
        genre_present = df['genres'] != '[]'

        # Series of the genres present in the movies_metadata
        genres = df['genres'][genre_present]

        # Step 2: Separating and selecting the genres
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(genres)
        label_classes = mlb.classes_
        label_data = pd.DataFrame(labels, columns=label_classes)
        selected_genres = label_data.sum(axis=0)
        selected_genres = selected_genres[selected_genres > 1].index
        final_genres = MultiLabelBinarizer(classes=selected_genres)
        y = final_genres.fit_transform(genres)

        # Including only the rows with labels present
        no_label_classes = y.sum(axis = 1) == 0
        y = y[~no_label_classes]
        X = df['overview'][~no_label_classes]
        return X, y, list(selected_genres)
