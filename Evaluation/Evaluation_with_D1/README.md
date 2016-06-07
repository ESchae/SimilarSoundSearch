# About this folder

This folder contains all files related to the evaluation of the algorithm
with the dataset 1 (D1) and the benchmark.

Algorithms compared:

* the complete algorithm with all extracted features (allfeatures)
* the same algorithm, but without the MFCC (withoutmfcc)
* the same algorithm, using only MFCC (onlymfcc)
* Freesound's algorithm (freesound)

Used evaluation measures:
* average score deviation
* score accuracy (deviations of 0, 1 or 2)
* nDCG

The R-Test was used to compute the significance.

Note: Files containing the Freesound distances between the queries and the 150 sounds
from D1 and the original distances within Freesound (including also sounds which are
not in D1) can be found in ../D1.

## Contents

* benchmark.txt --> the benchmark: it contains a line for every query of the ten queries from
D1, each line starts with the query id (Freesound), followed by a tab and then by the resultlist.
The resultlist contains all ids of the 149 remaining sounds in D1 together with their
similarity score (0-5), sorted by decreasing similarityscore. 

* resultlists_... .txt --> benchmark-like files, but with the resultlists from the four algorithms from above. Note that these contain distances instead of a similarity score and hence are sorted ascending by distance to the query sound

* evaluation_overview... .txt --> the results of the evaluation measures for every algorithm and query as well as the averages over all queries per algorithm

* scores.dat --> the non scales distances of the four algorithms for every query in increasing order, was used for plotting

* scores_scaled ... .dat --> dito, but the scaled versions

* significance_ ... .txt --> all pairwise p-values of the R-Test

* evaluation_utils.py --> code used for evaluation purposes
