import kaggle
from pathlib import Path
import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


def preprocess(movies_data_path):
    """
    Function to preprocess the movies dataset.
    Largely inspired to https://www.kaggle.com/neerajkhadagade/predicting-the-movie-genres.
    """

    # read the csv
    df = pd.read_csv(movies_data_path / 'movies_metadata.csv')
    # drop unnecessary columns
    df = df[['title', 'tagline', 'original_title', 'overview', 'genres']]
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
    selected_genres = selected_genres[selected_genres > 1]
    final_genres = MultiLabelBinarizer(classes=selected_genres.index)
    y = final_genres.fit_transform(genres)
    # Including only the rows with labels present
    no_label_classes = y.sum(axis = 1) == 0
    y = y[~no_label_classes]
    
    # STEP 3: create feature vectors
    X = df['overview'][~no_label_classes]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size = 0.3, random_state = 42)

    vectorizer = TfidfVectorizer(max_features = 1000, stop_words='english', lowercase=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_valid_vec = vectorizer.transform(X_valid)
    return X_train_vec, X_valid_vec, y_train, y_valid, [i for i in selected_genres.index], vectorizer

                                                                
if __name__ == "__main__":
    data_path = Path("./data")
    # data_path.mkdir(parents=True, exist_ok=True)
    # kaggle.api.authenticate()
    # kaggle.api.dataset_download_files('rounakbanik/the-movies-dataset', path=data_path, unzip=True)
    # unused_files = [i for i in data_path.iterdir() if i != data_path / "movies_metadata.csv"]
    # for f in unused_files:
    #     f.unlink()
    X1, X2, y1, y2, genre_counts, vect = preprocess(data_path)

    ### Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(X1, y1)
    train_pred = model.predict(X1)
    valid_pred = model.predict(X2)

    print("Classification Report")
    print("Training:\n", classification_report(y_true=y1, y_pred=train_pred, target_names=genre_counts))
    print("Validation:\n", classification_report(y_true=y2, y_pred=valid_pred, target_names=genre_counts))

    print("Accuracy")
    train_rfc_acc = accuracy_score(y_true=y1, y_pred=train_pred)
    valid_rfc_acc = accuracy_score(y_true=y2, y_pred=valid_pred)
    print("Traning: ", train_rfc_acc)
    print("Validation: ",valid_rfc_acc)

    # save
    joblib.dump(model, data_path / "model.pkl")
    joblib.dump(vect, data_path / "vect.pkl")



