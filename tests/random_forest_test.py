import pytest
from movie_classifier import RandomForest
PATH = '.'

def test_invalid_max_features():
    with pytest.raises(ValueError):
        RandomForest(PATH, max_features=0)


@pytest.mark.parametrize('size', [-1, 2])
def test_test_size(size):
    with pytest.raises(ValueError):
        RandomForest(PATH, test_size=size)


def test_load():
    # TODO: add test for keys in dictionary
    obj = RandomForest(PATH)
    with pytest.raises(FileNotFoundError):
        obj.load()


def test_train_features_targets():
    features = [0] * 2
    targets = [0] * 3
    genres = ['temp']
    obj = RandomForest(PATH)
    with pytest.raises(ValueError):
        obj.train(features, targets, genres)


def test_train_features_less_genres():
    features = [0] * 2
    targets = [0] * 2
    genres = [0] * 3
    obj = RandomForest(PATH)
    with pytest.raises(RuntimeError):
        obj.train(features, targets, genres)


def test_train_no_genres():
    features = [0] * 5
    targets = [0] * 5
    genres = []
    obj = RandomForest(PATH)
    with pytest.raises(RuntimeError):
        obj.train(features, targets, genres)


def test_predict_title():
    obj = RandomForest(PATH)
    with pytest.raises(TypeError):
        obj.predict(1, 'hello')


def test_predict_description():
    obj = RandomForest(PATH)
    with pytest.raises(TypeError):
        obj.predict('hello', 1)


def test_predict_model():
    obj = RandomForest(PATH)
    with pytest.raises(RuntimeError):
        obj.predict('hello', 'world')
