# SimilarSoundSearch
A small command line query-by-example search engine for similar sounds.

This repository contains the code and evaluation data related to a bachelor's
thesis written at the University of Freiburg. A content-based search algorithm
for similar sounding sounds was implemented and evaluated using crouwdsourcing.

Given an audiofile as query and a specific database with sounds the algorithm returns
the most similar sounds to the query. You can directly listen to the sounds from 
the terminal.

## Usage

To perform a search you need to call similarsounds.py from the terminal.
Given a soundbase and a query the program will return a list
of most similar sounds to the query found in the soundbase. If no query is specified, a randomly selected sound from the soundbase will be used.
Once presented with the results you can decide if you want to listen to the sounds directly from the terminal, perform a new query or exit.


![alt tag](https://cdn.rawgit.com/ESchae/SimilarSoundSearch/master/usage.gif)



If you want to customize your search check

`$ python similarsounds.py -h`

for more options.


## Requirements

### Linux

* python 2.x (used version is 2.7.6., theres no support yet for python 3 because of missing support for python 3 within Essentia (see https://github.com/MTG/essentia/issues/138)

* [Essentia](http://essentia.upf.edu/) (v.2.1._beta2 _fixes) for feature extraction.
You can download it from [here](https://github.com/MTG/essentia/tree/v2.1_beta2_fixes) for a complete installation follow the instructions [here](http://essentia.upf.edu/documentation/installing.html).
For SimilarSoundSearch it should be enough to do the following:

Install the dependencies
`sudo apt-get install build-essential libfftw3-dev libavcodec-dev libavformat-dev libavutil-dev libavresample-dev python-dev libsamplerate0-dev python-numpy-dev python-numpy`

Then go into the source directory and configure with
`./waf configure --mode=release --with-python`
compile
`./waf`
and insall (you might need sudo).
`./waf install`


Additional python packages:

* [scikit-learn](http://scikit-learn.org/stable/install.html) for nearest neighbors search (requires numpy and scipy)

* [dataset](http://dataset.readthedocs.io/en/latest/install.html) 0.6.0 for sql workarounds

* [tqdm](https://pypi.python.org/pypi/tqdm) for displaying a nice progressbar 

`$ pip install scikit-learn dataset tqdm`



For listen to audiofiles directly from terminal:

* [SoX](https://wiki.ubuntuusers.de/SoX/) for listen to the files directly from the terminal
and a handler for mp3 files

`$ sudo apt-get install sox libsox-fmt-mp3`


If using the example database is not enough for you and you want to build your own just try

`$ python soundbase.py -h`


Up to now the functionality was only tested on Ubuntu 14.04.




