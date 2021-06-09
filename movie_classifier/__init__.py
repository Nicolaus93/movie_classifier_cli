"""
`movie_classifier` is a package implementing
a command line interface used to classify
movies based on title and description.
"""
from .cli import CLInterface
from .dataset import KaggleDataset
from .random_forest import RandomForest

__all__ = ["CLInterface", "KaggleDataset", "RandomForest"]
__version__ = "0.1.0"
