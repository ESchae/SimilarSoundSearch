"""
Copyright 2016 University of Freiburg
Elke Schaechtele <elke.schaechtele@web.de>

This module is used to preform the search using a k-Nearest-Neighbors
implementation from scikit learn, see
http://scikit-learn.org/stable/modules/neighbors.html

"""
from __future__ import print_function
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from feature_extractor import FeatureExtractor
from soundbase import SoundBase
import time
import sys
import pickle
import numpy as np
import logging
import os

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Search():
    """ Class for performing similar sound search.

    Search will be performed with the help of a unsupervised NearestNeighbor
    algorithm implemented in python library scikit.learn, see
    http://scikit-learn.org/stable/index.html
    To build the search space a soundbase must be specified as implemented
    in class SoundBase. Additionally a list of features can be specified to
    determine which audiofeatures should be included in the feature vectors
    from which the search space will be build. To perform a search, an instance
    must be called with the method query which takes an audiofile as argument.
    It is possible to save the computed search space for future use, i.e.
    to make the data persist after program termination.


    Attributes
    ----------
        n (int): number of nearest neighbors to compute
        algorithm (str): the underlying algorithm used for Nearest Neighbors
        metric (str): distance metric used to calculate the similarity
        soundbase (soundbase): the sql database containing sounds
            in which is seached for similar sounds
        scaler (sklearn.preprocessing.data.StandardScaler): the scaler which
            performs scaling of the sounds
        features (list[str]): names of features which are taken into account
        nbrs (sklearn.neighbors.unsupervised.NearestNeighbors): the search
            space on which queries will be processed
    """

    def __init__(self, soundbase, features=[], stats=[], n=5,
                 algorithm='auto', metric='euclidean',
                 nbrs_input=None, scaler_input=None,
                 nbrs_output=None, scaler_output=None):
        """ Instantiates the nearest NeirestNeighbors search space.

        If no features are specified, all features available in
        Features class will be used. If the soundbase contains no information
        about extracted features it is impossible to build the search and an
        exception will be thrown.
        It is possible to save the data for future use, by specifying
        filenames for nbrs_output and scaler_output

        Parameters
        ----------
            soundbase (soundbase): a sql database containing sounds
                (aka SoundBase)
            features (optional(list[str])): names of features which are taken
                into account, default='', which means all features will be used
                available features: ['pitch', 'pitchConfidence', 'mfcc',
                'spectralcentroid', 'duration', 'logattacktime',
                'dynamicComplexity', 'effectiveDuration', 'loudness']
                Note: the available features listed here assume the default
                setting of FeatureExtractor class was used for extraction
            stats (optional(list[str])): which of the aggregated feature
                values should be used, possible options: ['mean', 'var',
                'dmean', 'dvar'], default='', which means all are used
                Note: the available features listed here assume the default
                setting of FeatureExtractor class was used for extraction
            n (int): number of nearest neighbors to return, default=5
            algorithm (optional(str)): determines the underlying algorithm
                of the neighbors search algorithm from scikit.learn,
                options: ['auto', 'brute', 'ball_tree', 'kd_tree'],
                if 'auto' the algorithm tries to determine the best approach
                from the training data, 'brute' corresponds to pairwise
                comparisions, tree algorithms are more efficient for soundbases
                containing a lot of samples, default='brute'
            metric (optional(str)): the metric used for the distance
                computations, for a list of available metrics see
                documentation of class DistanceMetric in scikit.learn,
                default='cosine'
            nbrs_input (optional(str)): filename of a saved search space which
                should be used instead of computing a new one
            scler_input (optional(str)): filename of a saved scaler, for using
                a saved search space it is neccesary to indicate a scaler
                as well
            nbrs_output (optional(str)): filename for data persistance of
                search space
            scaler_output (optional(str)): filename for data persistance of
                scaler

        Returns
        -------
            None


        Examples
        --------

        >>> features = ['spectralcentroid']
        >>> stats = ['var', 'mean']
        >>> S = Search('Testfiles/test.db', features=features, stats=stats)
        >>> S.features
        ['spectralcentroid_var', 'spectralcentroid_mean']
        >>> S.metric
        'euclidean'
        >>> S.algorithm
        'auto'
        >>> S = Search('Testfiles/empty.db')
        Traceback (most recent call last):
         ...
        IOError: No features found in soundbase

        """
        self.n = n
        self.algorithm = algorithm
        self.metric = metric

        # connect to soundbase
        if not os.path.exists(soundbase):
            raise IOError('SoundBase %s does not exist' % soundbase)
        self.soundbase = SoundBase(soundbase)
        if self.soundbase.list_features_in_database() is None:
            raise IOError('No features found in soundbase')

        # initialize scaler needed for scaling the feature vectors equally
        if scaler_input:
            logger.info('Loader scaler from %s ' % scaler_input)
            with open(scaler_input) as f:
                data = f.read()
            self.scaler = pickle.loads(data)
        else:
            self.scaler = preprocessing.StandardScaler()
        self.features = None
        self.nbrs = None

        # for data persistance nbrs and scaler is needed
        if nbrs_input and not scaler_input:
            logging.warning('No scaler could be found therefore search space \
is unusable because a query could not be scaled according to the original \
scaling. Search space will be recomputed.')
            nbrs_input = None

        if nbrs_output and not scaler_output:
            logging.warning('Saving search space without scaler makes search \
unusable for the future because a new query could not be scaled acoording \
to the used scaling.')

        # up tp now it is only advised to make data persistant within
        # the default settings
        if nbrs_output and (features or stats):
            logger.warning('You are saving a search space which does not \
include the default feature settings. This will not be checked if you want \
to reuse the search space the next time. You have to be sure yourself that \
you use the same setting as the last time otherwise unexpected things might \
happen.')
        if nbrs_input and (features or stats):
            logger.warning('You are now using a saved search space with a \
feature setting that is not the default one. Be sure you know what you are \
doing becuase there is no assurance that the feature space was saved with the \
same setting as the one you are using now.')

        # load search space or compute it new
        if nbrs_input:
            logger.info('Loading search space from %s' % nbrs_input)
            start_time = time.time()
            with open(nbrs_input) as f:
                data = f.read()
            self.nbrs = pickle.loads(data)
            end_time = time.time()
            logger.info('Done in %.2f s' % (end_time - start_time))
            self.set_features(features, stats, build=False,
                              nbrs_output=nbrs_output,
                              scaler_output=scaler_output)
        else:
            self.set_features(features, stats, build=True,
                              nbrs_output=nbrs_output,
                              scaler_output=scaler_output)

    def set_features(self, features=[], stats=[], build=True,
                     nbrs_output=None, scaler_output=None):
        """ Sets the features to be used for searching.

        This is a workaround to be able to set features easier.
        Instead of naming e.g. pitch_mean, pitch_var, mfcc_mean, mfcc_var ...
        exactly, it is possible to name pitch and mfcc as features
        and mean and var as stats.
        Be aware that if new features are set the searchspace must be
        computed new.

        Parameters
        ----------
            features (optional(list[str])): names of features which are taken
                into account, default='', which means all features will be used
                available features: ['pitch', 'pitchConfidence', 'mfcc',
                'spectralcentroid', 'duration', 'logattacktime',
                'dynamicComplexity', 'effectiveDuration', 'loudness']
                Note: the available features listed here assume the default
                setting of FeatureExtractor class was used for extraction
            stats (optional(list[str])): which of the aggregated feature
                values should be used, possible options: ['mean', 'var',
                'dmean', 'dvar'], default='', which means all are used
                Note: the available features listed here assume the default
                setting of FeatureExtractor class was used for extraction
            build (optional(bool)): whether to compute the search space new,
                default=True
            nbrs_output (optional(str)): filename for data persistance of
                search space
            scaler_output (optional(str)): filename for data persistance of
                scaler

        Returns
        -------
            None


        Examples
        --------
        >>> S = Search('Testfiles/test.db')
        >>> S.features # doctest: +NORMALIZE_WHITESPACE
        ['pitchConfidence_dvar', 'mfcc_mean', 'spectralcentroid_dvar',
        'duration', 'mfcc_dvar', 'pitchConfidence_var', 'logattacktime',
        'pitch_dmean', 'dynamicComplexity', 'spectralcentroid_dmean',
        'spectralcentroid_var', 'pitchConfidence_mean', 'pitch_dvar',
        'pitch_mean', 'mfcc_dmean', 'pitch_var', 'mfcc_var',
        'pitchConfidence_dmean', 'spectralcentroid_mean',
        'effectiveDuration', 'loudness']

        >>> S.set_features(['mfcc', 'spectralcentroid'])
        >>> S.features # doctest: +NORMALIZE_WHITESPACE
        ['mfcc_dvar', 'mfcc_mean', 'mfcc_var', 'mfcc_dmean',
         'spectralcentroid_dvar', 'spectralcentroid_mean',
         'spectralcentroid_var', 'spectralcentroid_dmean']

        >>> S.set_features(['mfcc', 'logattacktime'], stats=['mean', 'var'])
        >>> S.features
        ['mfcc_mean', 'mfcc_var', 'logattacktime']

        """
        self.features = []
        aggr_features, avail_stats, global_features = self.available_features()
        start_time = time.time()
        if not features and not stats:  # include all extracted features
            logging.info('Using all features')
            self.features = self.soundbase.list_features_in_database()
        elif not features and stats:   # not yet supported
            logging.warning('Stats specified but no features')
        else:
            for feature in features:
                if feature in global_features:
                    self.features.append(feature)  # no stats here
                elif feature in aggr_features:
                    if not stats:
                        stats = avail_stats  # include all stats
                    for stat in stats:
                        if stat not in avail_stats:
                            logging.warning('%s was not used for aggregation' %
                                            stats)
                            continue
                        self.features.append(feature + '_' + stat)
                else:
                    logging.warning('Feature %s not in soundbase' % feature)
        end_time = time.time()
        logger.info("Features set in %.2f s. Features: %s"
                    % ((end_time - start_time), self.features))
        # rebuild NearestNeighbors search space
        if build:
            self._build(nbrs_output=nbrs_output, scaler_output=scaler_output)

    def _build(self, nbrs_output=None, scaler_output=None):
        """ Method for building the Nearest Neighbor search space.

        Parameters
        ----------
            nbrs_output (optional(str)): filename for data persistance of
                search space
            scaler_output (optional(str)): filename for data persistance of
                scaler

        Returns
        -------
            None

        """
        # get samples
        samples = self._neighbors_samples(scaler_output=scaler_output)
        # build NearestNeighbors search space
        logger.info("Building Nearest Neighbor")
        start_time = time.time()
        self.nbrs = NearestNeighbors(n_neighbors=self.n,
                                     algorithm=self.algorithm,
                                     metric=self.metric).fit(samples)
        end_time = time.time()
        logger.info("Done in %2f s" % (end_time - start_time))
        # write to file for data persistence if desired
        if nbrs_output:
            with open(nbrs_output, 'w') as f:
                f.write(pickle.dumps(self.nbrs))

    def available_features(self, print_output=False):
        """ Returns a nice overview over available framebased and global
        features as well as the statistics used to aggregate framebased
        features. This method is used for checking if a feature or statistic
        was computed and hence exists in the database before it will be set in
        method set_features

        Parameters
        ----------
            print_output (optional(bool)): if True, the output will be pretty
                printed, otherwise features are returned

        Returns
        -------
        if not print_output:
            aggregated_features, stats, global_features (list[list[ints]]):
                the available aggregated and global features as well as the
                available stistics for aggregation of framebased features
        else:
            None


        Examples
        --------
        >>> S = Search('Testfiles/test.db')
        >>> aggregated_features, stats, globl_features = S.available_features()
        >>> aggregated_features # doctest: +NORMALIZE_WHITESPACE
        ['pitchConfidence', 'mfcc', 'spectralcentroid', 'pitch']
        >>> stats
        ['dvar', 'mean', 'var', 'dmean']
        >>> globl_features # doctest: +NORMALIZE_WHITESPACE
        ['duration', 'logattacktime', 'dynamicComplexity',
        'effectiveDuration', 'loudness']

        >>> S.available_features(print_output=True) # doctest:
        ...    +NORMALIZE_WHITESPACE
        Framebased features:
            pitchConfidence
            mfcc
            spectralcentroid
            pitch
        Used statistics for aggregation: dvar mean var dmean
        Global features:
            duration
            logattacktime
            dynamicComplexity
            effectiveDuration
            loudness

        """
        all_features = self.soundbase.list_features_in_database()
        if not all_features:
            raise ValueError('No features found in soundbase')
        aggregated_features = []
        global_features = []
        stats = []
        for feature in all_features:
            try:
                aggregated_feature, stat = feature.split('_')
                if aggregated_feature not in aggregated_features:
                    # str used to prevent python to return a unicode string
                    aggregated_features.append(str(aggregated_feature))
                if stat not in stats:
                    stats.append(str(stat))
            except ValueError:  # split only revealed one value
                global_features.append(feature)
        if not print_output:
            return aggregated_features, stats, global_features
        # print available features nicely
        print('Framebased features:\n\t%s' % '\n\t'.join(aggregated_features))
        print('Used statistics for aggregation: %s' % " ".join(stats))
        print('Global features:\n\t%s' % '\n\t'.join(global_features))

    def _neighbors_samples(self, scaler_output=None):
        """ Helpermethod for collecting the samples to build NearestNeighbors.

        A single sample is the feature vector for one sound in the
        soundbase consisting only of the during initialization specified
        features that should be included.

        Returns
        --------
            samples (array): an array with shape
                (n_sounds, n_flattened_features) containing the arrays
                of the feature vectors
            scaler_output (optional(str)): filename for data persistance of
                scaler


        Examples
        --------
        >>> features = ['spectralcentroid']
        >>> stats = ['var', 'mean']
        >>> S = Search('Testfiles/test.db', features=features, stats=stats)
        >>> samples = S._neighbors_samples()
        >>> print(np.round(samples, 3))
        [[-0.599 -0.593]
         [-0.598 -0.554]
         [ 1.972 -0.34 ]
         [-0.172 -0.505]
         [-0.603  1.993]]

        """
        # get feature vectors for all sounds in soundbase
        samples = []
        logger.info("Collecting feature vectors...")
        start_time = time.time()
        for sound in self.soundbase.sounds:
            sample = self.feature_vector(sound, self.features,
                                         deserialize=True)
            samples.append(sample)
        end_time = time.time()
        logger.info("Done in %.2f s" % (end_time - start_time))
        samples = np.array(samples)
        # standardization of feature vectors
        logger.info("Standardizing samples")
        start_time = time.time()
        self.scaler = self.scaler.fit(samples)
        samples_standardized = self.scaler.transform(samples)
        end_time = time.time()
        logger.info("Done in %.2f s" % (end_time - start_time))
        # save scaler to file for future use if desired
        if scaler_output:
            with open(scaler_output, 'w') as f:
                f.write(pickle.dumps(self.scaler))
        return samples_standardized

    def feature_vector(self, extracted_features, features_to_include,
                       deserialize=False):
        """ Calculates a feature vector from a given features dict.

        Parameters
        ----------
            extracted_features (dict): contins all names of extracted features
                as keys with their corresponding values
            include_features (list[str]): a list specifying the names of
                features which should be included into the feature vector

        Returns
        -------
            feature_vector (array[floats]): a numpy array containing the
                flattened and deserialized (if needed) values of the given
                features


        Examples
        --------
        >>> extrf = {'a': 2, 'b': 3, 'c': np.array([0,1]),
        ...    'd': np.array([4,5])}
        >>> inclf = ['b', 'c', 'd']
        >>> Search('Testfiles/test.db').feature_vector(extrf,
        ...    inclf, deserialize=False)
        array([3, 0, 1, 4, 5])

        """
        if features_to_include is None:
            logger.warning("""No features found which should be \
included in feature vector""")
            return None
        if extracted_features is None:
            logger.warning("""No extracted features found from which the \
feature vector could be build""")
            return None
        if deserialize:
            feature_vector = np.hstack(
                (pickle.loads(extracted_features[feature])
                 for feature in features_to_include))
        else:
            feature_vector = np.hstack(
                (extracted_features[str(feature.replace('_', '.'))]
                 for feature in features_to_include))
        return feature_vector

    def query(self, audiofile):
        """ Method for processing a query on a Search instance.

        Note that if the query exists in the soundbase it will be at the first
        position in the result list.

        Parameters
        ----------
            audiofile (str): filename of the query audiofile

        Returns
        -------
            distances (array[floats]), indices (array[floats]):
                the n nearest neighbors found to the query,
                specified via their indices and distances to the query,
                the indices correspond to the id - 1 of the sounds in the
                soundbase


        Examples
        --------
        >>> S = Search('Testfiles/test.db')
        >>> distances, ids = S.query('Testfiles/test_short.wav')
        >>> np.round(distances, 2)
        array([[  0.  ,  14.  ,  14.01,  15.3 ,  16.24]])
        >>> ids
        array([[4, 1, 0, 3, 2]])

        """
        # get features and feature vector from query
        features = FeatureExtractor(audiofile).features
        query_vector = self.feature_vector(features, self.features)
        # scale query according to the scaling used for scaling the
        # feature vectors of the soundbase
        query_vector = self.scaler.transform([query_vector])
        # compute k nearest neighbors
        logger.info('Calculating %d most similar sounds for %s' %
                    (self.n, audiofile))
        result = self.nbrs.kneighbors(query_vector[0])
        return result

    def print_result(self, result, output=sys.stdout):
        """ Method for printing the query result nicely.

        Parameters
        ----------
            result (array) as returned by method query


        Examples
        --------
        >>> s = Search('Testfiles/test.db')
        >>> result = s.query('Testfiles/test_short.wav')
        >>> s.print_result(result, output=None)
        NN 1: Testfiles/test_short.wav with distance 0.00
        NN 2: Testfiles/sine440.wav with distance 14.00
        NN 3: Testfiles/sine300.wav with distance 14.01
        NN 4: Testfiles/test_recurse/test.wav with distance 15.30
        NN 5: Testfiles/sine880.mp3 with distance 16.24
        >>> s.print_result('') is None
        True

        """
        if result is '' or result is None:
            logger.info("Can not print result because no result is given")
            return None
        dist, inds = result
        for i, neighbor in enumerate(inds[0]):
            row = self.soundbase.get_row(neighbor + 1)
            f = row['path']
            print("NN %d: %s with distance %.2f" % (i + 1, f, dist[0][i]),
                  file=output)

    def _get_freesound_ids(self, result_ids):
        """ Method for retrieving the freesound id of a given list of results
        for a query of D1.

        This method is used for evaluation and serves as helpermethod for
        method _results. The freesound ids are needed for building a
        benchmark-like file with all result lists of the ten queries from D1.

        Parameters
        ----------
            result_ids (list[int]): a list of indices, each index is the
                index of a row in the nearest neighbors samples

        Returns
        -------
            freesound_ids (list[int]): the corresponding freesound ids to the
                indices


        Examples
        --------
        >>> S = Search('example.db')
        >>> dists, ids = S.query('Evaluation/D1/audiofiles/0-110011.mp3')
        >>> result_ids = ids[:6][0]
        >>> result_ids
        array([  0,  10,  20, 119,  12])
        >>> S._get_freesound_ids(result_ids)
        [110011, 256452, 315834, 83930, 315829]

        """
        freesound_ids = []
        for result_id in result_ids:
            # soundbase index counting starts at 1
            path = self.soundbase.get_row(result_id + 1)['path']
            path_without_ext = path[:-4]  # all end with .mp3
            freesound_id = int(path_without_ext.split('-')[-1])
            freesound_ids.append(freesound_id)
        return freesound_ids

    def _results(self, query_ids):
        """ Helpermethod for retrieving ids and distances for several queries.

        This method is intended to be used for evaluating the algorithm with
        the benchmark generated with crowdflower.
        The ids in the result list will be the freesound ids.
        Distances will be rounded to two decimals.

        Examples
        --------
        >>> S = Search('example.db', n=3, features='')
        >>> query_ids = ['Evaluation/D1/audiofiles/0-2155.mp3',
        ...    'Evaluation/D1/audiofiles/0-5560.mp3']
        >>> results = S._results(query_ids)
        >>> results[5560]
        [(5560, 0.0), (17786, 3.75), (214052, 3.84)]

        """
        results = dict()
        # add result_ids and distances for every query
        for query in query_ids:
            dists, ids = self.query(query)  # they are already sorted
            ids = self._get_freesound_ids(ids[0])
            dists = np.round(dists[0], 2).tolist()
            result_list = zip(ids, dists)
            query = int(query.split('-')[1][:-4])  # freesound id of query
            results[query] = result_list
        return results

    def _fileoutput_for_evaluation(self, filename='algorithm_results.txt',
                                   query_ids=['data/0-110011.mp3',
                                              'data/0-2155.mp3',
                                              'data/0-22362.mp3',
                                              'data/0-240015.mp3',
                                              'data/0-264498.mp3',
                                              'data/0-325462.mp3',
                                              'data/0-337791.mp3',
                                              'data/0-50802.mp3',
                                              'data/0-5560.mp3',
                                              'data/0-82402.mp3']):
        """ Method for writing results for several queries to a file.

        The file is used for evaluation. This function is intended to be
        used for evaluating the algorithm with the benchmark generated
        with crowdflower.
        The first results is the query itself, hence it will be not included
        in the file output.
        """
        results = self._results(query_ids)
        with open(filename, 'w') as f:
            for query, result_list in results.items():
                query_row = '%d\t' % int(query)
                for result, score in result_list:
                    if result == query:
                        continue  # query itself should not be included
                    query_row += '%d=%.2f ' % (result, score)
                query_row += '\n'
                f.write(query_row)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
