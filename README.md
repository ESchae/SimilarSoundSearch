# SimilarSoundSearch
A small command line query-by-example search engine for similar sounds.

This repository contains the code and evaluation data related to a bachelor's
thesis written at the University of Freiburg. A content-based search algorithm
for similar sounding sounds was implemented and evaluated using crouwdsourcing.

Given an audiofile as query and a specific database with sounds the algorithm returns
the most similar sounds to the query. You can directly listen to the sounds from 
the terminal.

## Requirements

* python 2.x (used version is 2.7.6., theres no support yet for python 3 because of missing support for python 3 within Essentia (see https://github.com/MTG/essentia/issues/138)

The following additional libraries must be installed:


* For feature extraction Essentia (branch 2.1._beta2 _fixes) is used.
You can download it from [here](https://github.com/MTG/essentia) and follow the instructions given[here](http://essentia.upf.edu/documentation/installing.html)

* [scikit-learn](http://scikit-learn.org/stable/install.html) for nearest neighbors search (requires numpy and scipy)

> pip install scikit-learn


For working with the SoundBase class you need the following:

* [dataset](http://dataset.readthedocs.io/en/latest/install.html) 0.6.0 for sql workarounds

* [tqdm](https://pypi.python.org/pypi/tqdm) for displaying a nice progressbar 

> pip install dataset tqdm



Up to now the functionality was only tested on Ubuntu 14.04.
