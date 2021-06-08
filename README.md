# Movie classifier

This repository contains a package implementing a command line interface used to classify movies given a title and a description in input. To install the package run
``` 
python setup.py install
```

## Usage

Once installed, you can use the package by invoking the command `movie_classifier` from the command line and providing `--title` and `--description` arguments, i.e.
```
movie_classifier --title 'Othello' --description 'The evil Iago pretends to be friend of Othello in order to manipulate him to serve his own end in the film version of this Shakespeare classic.'
```
The first time running this command, the package will download the data necessary to train the model and subsequently train the a random forest model.

Later, any other time the command is invoked the model will be loaded and used directly for prediction.