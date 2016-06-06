"""
Copyright 2016
Author: Elke Schaechtele <elke.schaechtele@web.de>

This module is used for evaluating the algorithm.
"""
from __future__ import print_function
import math
import logging
import sys

logging.basicConfig(level=logging.WARNING)


class Evaluation():
    """ Class for evaluating the implemented similarity search algorithm.

    An instance of the class must be instantiated with a benchmarkfile.
    The benchmark mus be in the style of the one given in this folder.
    This means each line starts with a query id, followed by a tab and then
    followed by the result list. The result list contains space separated
    id=score pairs.
    Evaluation will be run on a evaluation file.
    This file must have the same form as the benchmark.

    Per default the following evaluation measure are used:

    - average score deviation
    - accuracy (d=0, d=1 and d=2)
    - nDCG

    It is possible to scale scores linear or logarithmic.

    """

    def __init__(self, benchmark_file):
        """ Intitialize from a given benchmark file.

        Parameters
        ----------
            benchmark_file (str): filename of the benchmark, the benchmark
                must have a specific form (see class descriptions)

        Examples
        --------
        >>> E = Evaluation('../Testfiles/test_benchmark.txt')
        >>> sorted(E.benchmark_ids.items())
        [(1, [2, 4, 5, 3, 6, 8, 7]), (2, [1, 6, 4, 3, 8, 7, 5])]
        >>> sorted(E.benchmark_scores.items()) # doctest: +NORMALIZE_WHITESPACE
        [(1, [3.0, 3.0, 3.0, 2.0, 1.0, 1.0, 0.0]), \
         (2, [5.0, 5.0, 3.0, 3.0, 3.0, 1.0, 0.0])]

        """
        self.benchmark = benchmark_file
        logging.info('Using %s as benchmark' % self.benchmark)
        self.benchmark_ids, self.benchmark_scores = self.get_ids_and_scores(
            self.benchmark)

    def get_ids_and_scores(self, filename):
        """ Method for reading a benchmark-like file for evaluation.

        The method extracts for every query in the file the result list and
        splits it into a score and a result_id list. Both lists will be sorted
        by score. Then two dicts will be build: A score dict and an id dict.
        Each dict contains the queries as keys and the score (resp. id) lists
        as values for each query.

        Parameters
        ----------
            filename (str): the name of the file from which the scores and ids
                should be extracted. This can be either a benchmark or a file
                containing the results of an algorithm which should be
                evaluated

        Returns
        -------
            ids, scores (dicts) : the dicts containing id and score information
                as described above

        Examples
        --------
        >>> E = Evaluation('../Testfiles/test_benchmark.txt')
        >>> testfile = '../Testfiles/test_evaluation_file.txt'
        >>> ids, scores = E.get_ids_and_scores(testfile)
        >>> sorted(ids.items())
        [(1, [7, 2, 4, 3, 5, 6, 8]), (2, [6, 1, 4, 3, 8, 7, 5])]
        >>> sorted(scores.items()) # doctest: +NORMALIZE_WHITESPACE
        [(1, [5.0, 4.0, 3.0, 3.0, 1.0, 1.0, 1.0]), \
         (2, [5.0, 4.0, 3.0, 3.0, 3.0, 1.0, 0.0])]

        """
        ids = dict()
        scores = dict()
        with open(filename) as f:
            results = f.readlines()
        for line in results:
            query, result_list = line.split('\t')
            query = int(query)
            for i, d in enumerate([ids, scores]):
                for result in result_list.split():
                    result_id, score = result.split('=')
                    if not d.get(query):
                        d[query] = []
                    d[query].append(int(result_id) if (i == 0)
                                    else float(score))
        return ids, scores

    def get_relevance_scores(self, query, id_list):
        """ Helpermethod for retrieving the benchmark scores for a given
        id list.

        The result list is meant to stem from an algorithm.
        Used for computing nDCG.

        Parameters
        ----------
            query (int): the query for wich the relevance scores should be
                retrieved
            id_list (list[int]): the result list of the query containing
                all retrieved ids sorted by the relevance computed by the
                algorithm

        Returns
        -------
            relevance_scores (list[int]): a list containing the relevance
                score for each result in the id_list as given in the benchmark

        Examples
        --------
        >>> E = Evaluation('../Testfiles/test_benchmark.txt')
        >>> testfile = '../Testfiles/test_evaluation_file.txt'
        >>> ids, scores = E.get_ids_and_scores(testfile)
        >>> E.get_relevance_scores(1, ids[1])
        [0.0, 3.0, 3.0, 2.0, 3.0, 1.0, 1.0]
        >>> E.get_relevance_scores(2, ids[2])
        [5.0, 5.0, 3.0, 3.0, 3.0, 1.0, 0.0]

        """
        relevance_scores = []
        # add the benchmark score for every result in the order they appear
        for id_ in id_list:
            try:
                position_in_benchmark = self.benchmark_ids[query].index(id_)
                score = self.benchmark_scores[query][position_in_benchmark]
                relevance_scores.append(score)
            except ValueError:
                logging.warning("%d not contained in the benchmarkresult of %d"
                                % (id_, query))
                logging.warning(len(id_list))
        return relevance_scores

    def compute_average_score_diff(self, query, id_list, score_list, k=14):
        """ Method for computing the average score difference for a query.

        Parameters
        ----------
            query (int): the query for which to compute the measure
            id_list (list[int]): the ids of the result list of the query
            score_list (list[float]): the scores of the result list of the
                query
            k (optional[int]): the rank up to which the average score
                difference should be computed

        Returns
        -------
            average score difference (float)

        Examples
        --------
        >>> E = Evaluation('../Testfiles/test_benchmark.txt')
        >>> testfile = '../Testfiles/test_evaluation_file.txt'
        >>> zip(E.benchmark_ids[1], E.benchmark_scores[1])
        [(2, 3.0), (4, 3.0), (5, 3.0), (3, 2.0), (6, 1.0), (8, 1.0), (7, 0.0)]
        >>> ids, scores = E.get_ids_and_scores(testfile)
        >>> zip(ids[1], scores[1])
        [(7, 5.0), (2, 4.0), (4, 3.0), (3, 3.0), (5, 1.0), (6, 1.0), (8, 1.0)]
        >>> asd = 9.0 / 7.0
        >>> asd == E.compute_average_score_diff(1, ids[1], scores[1])
        True

        """
        if not k:
            k = len(id_list)  # evaluate complete list
        score_diff = 0
        # get respective score from benchmark for every id
        benchmark_scores = self.get_relevance_scores(query, id_list)
        for i, score in enumerate(score_list[:14]):
            score_diff += abs(benchmark_scores[i] - score_list[i])
        return score_diff / len(score_list[:k])

    def compute_accuracy(self, query, id_list, score_list, k=None,
                         deviation=2):
        """ Method for computing the accuracy at deviation d for a query.

        Parameters
        ----------
            query (int): the query for which to compute the measure
            id_list (list[int]): the ids of the result list of the query
            score_list (list[float]): the scores of the result list of the
                query
            k (optional[int]): the rank up to which the accuracy should be
                computed
            deviation (int): the numbers of deviations to include in the
                accuracy computation

        Returns
        -------
            accuracy at deciation d (float)

        Exammples
        ---------
        >>> E = Evaluation('../Testfiles/test_benchmark.txt')
        >>> testfile = '../Testfiles/test_evaluation_file.txt'
        >>> zip(E.benchmark_ids[1], E.benchmark_scores[1])
        [(2, 3.0), (4, 3.0), (5, 3.0), (3, 2.0), (6, 1.0), (8, 1.0), (7, 0.0)]
        >>> ids, scores = E.get_ids_and_scores(testfile)
        >>> zip(ids[1], scores[1])
        [(7, 5.0), (2, 4.0), (4, 3.0), (3, 3.0), (5, 1.0), (6, 1.0), (8, 1.0)]
        >>> acc_0, acc_1, acc_2 = (3.0 / 7.0, 5.0 / 7.0, 6.0 / 7.0)
        >>> acc_0 == E.compute_accuracy(1, ids[1], scores[1], deviation=0)
        True
        >>> acc_1 == E.compute_accuracy(1, ids[1], scores[1], deviation=1)
        True
        >>> acc_2 == E.compute_accuracy(1, ids[1], scores[1], deviation=2)
        True

        """
        benchmark_scores = self.get_relevance_scores(query, id_list)
        acc = accuracy(score_list[:k], benchmark_scores[:k], deviation)
        return acc

    def compute_nDCG(self, query, result_list, k=None):
        """ Method for computing the nDCG of a given result list.

        The given result list is meant to stem from an algorithm.
        It will be used to compute the nDCG from the respective scores
        within the benchmark.

        Parameters
        ----------
            query (int): the query for which to compute the nDCG
            result_list (list[int]): a list containing the ids as given in
                the result list of the algorithm
            k (int): the rank up to which to compute the nDCG

        Examples
        --------
        >>> E = Evaluation('../Testfiles/test_benchmark.txt')
        >>> testfile = '../Testfiles/test_evaluation_file.txt'
        >>> ids, scores = E.get_ids_and_scores(testfile)
        >>> E.compute_nDCG(1, ids[1])
        0.816438598423691
        >>> E.compute_nDCG(2, ids[2])
        1.0

        """
        if not k:
            k = len(result_list)  # evaluate complete list
        relevance_scores = self.get_relevance_scores(query, result_list)[:k]
        return nDCG(relevance_scores)

    def evaluate(self, file_to_be_evaluated, k=149, scale=True, deviations=3,
                 queries=[337791, 264498, 5560, 240015, 50802, 110011, 22362,
                          2155, 82402, 325462],
                 scaletype='lin'):
        """ Method for computing all evaluation results for a given file.

        Parameters
        ----------
            file_to_be_evaluated (str): filename of the file which should
                be evaluated in comparion to the benchmark
                it must have the same structure as the benchmark
                (see class description)
            k (optional(int)): rank up to which the evaluation measures should
                be computed
            scale (optional(bool)): whether or not the scores have to be scaled
            deviations (optional(int)): the number of deviations to compute
                for the accuracy
            queries (list[int]): the ids of the queries in the evaluation file
            scaletype (str): how to scale the scores, possible options:
                'lin', 'log (default: 'lin')

        Returns
        -------
            resuts (list[floats]): a list containing all evaluation results
                for the query; at index 0 is the nDCG, followed by the accuracy
                values for the specified deviations, the last item is the
                average score deviation


        Examples
        --------
        >>> E = Evaluation('../Testfiles/test_benchmark.txt')
        >>> testfile = '../Testfiles/test_evaluation_file.txt'
        >>> results = E.evaluate(testfile, queries=[1, 2], scale=False)
        >>> sorted(results.items())  # doctest: +NORMALIZE_WHITESPACE
        [(1, [0.816438598423691, \
              0.42857142857142855, 0.7142857142857143, 0.8571428571428571, \
              1.2857142857142858]), \
         (2, [1.0, \
              0.8571428571428571, 1.0, 1.0, \
              0.14285714285714285])]

        """
        results = dict()
        ids, scores = self.get_ids_and_scores(file_to_be_evaluated)
        if scale:  # scale the scores to integer range 0-5
            scores = self.scale_scores(scores, scaletype=scaletype)
        logging.info("Evaluating %s:" % file_to_be_evaluated)
        for query in queries:
            try:
                query_results = []
                logging.info("Computing results for query %d" % query)
                # compute nDCG
                ndcg = self.compute_nDCG(query, ids[query], k=k)
                logging.info("nDCG_%d for query %d: %.2f" % (k, query, ndcg))
                query_results.append(ndcg)
                # compute accuracy at -deviations-
                for deviation in range(deviations):
                    acc = self.compute_accuracy(query, ids[query],
                                                scores[query], k=k,
                                                deviation=deviation)
                    logging.info("acc_%d for query %d: %.2f" %
                                 (deviation, query, acc))
                    query_results.append(acc)
                # compute average score difference
                asd = self.compute_average_score_diff(query, ids[query],
                                                      scores[query], k=k)
                logging.info("asd for query %d: %.2f" % (query, asd))
                query_results.append(asd)
                results[query] = query_results
            except KeyError:
                logging.warning("sound %d is not contained in benchmark"
                                % query)
        return results

    def scale_scores(self, score_dict, scaletype):
        """ Method for scaling all scores of a given algorithm.

        Parameters
        ----------
            score_dict (dict): a dict which contains queries as keys and the
                scores of their result lists as values

        Returns
        -------
            sclaed_scores (dict): a dict like the given one, but with scaled
                scores


        Examples
        --------
        >>> E = Evaluation('benchmark.txt')
        >>> ids, scores = E.get_ids_and_scores('resultlists_allfeatures.txt')
        >>> scores[2155][:10]
        [4.14, 4.57, 5.21, 5.41, 5.49, 5.74, 5.76, 5.8, 5.93, 6.01]
        >>> scaled_scores = E.scale_scores(scores, scaletype='lin')
        >>> scaled_scores[2155][:10]
        [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
        >>> scaled_scores = E.scale_scores(scores, scaletype='log')
        >>> scaled_scores[2155][:10]
        [3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]

        """
        scaled_scores = dict()
        # old_max = self.get_max_score(score_dict)
        for query, scores in score_dict.items():
            old_max = max(scores)
            scaled_scores[query] = []
            for score in scores:
                scaled_scores[query].append(
                    scale_score(score, new_range=5.0, old_min=0.0,
                                old_max=old_max, integer=True, reverse=True,
                                scaletype=scaletype))
        return scaled_scores

    def significance(self, files_to_be_compared, scaletype='lin'):
        """ Compute pairwise significance between the given files and print
        nicely to a file.

        This method is used to generate the files

        - significance_lin_scale.txt
        - significance_log_scale.txt


        Parameters
        ----------
            files_to_be_compared (list[str]): filenames of the agorithm
                results which should be evaluated
            scaletype (str): how to scale the scores ('lin' or 'log')

        Returns
        -------
            None

        """
        # get all evaluation results for the algorithms
        algorithm_results = []
        for algorithm in files_to_be_compared:
            algorithm_results.append(self.evaluate(algorithm,
                                                   scaletype=scaletype))
        measures = ['nDCG', 'acc_0', 'acc_1', 'acc_2', 'av']
        # get p-values from r-test
        with open('significance_%s_scale.txt' % scaletype, 'w') as f:
            for m, measure in enumerate(measures):
                f.write('p-value\t%s\n' % measure)
                for i, algorithm_result in enumerate(algorithm_results):
                    f.write(files_to_be_compared[i])
                    for j in range(4):
                        observations_a = [x[1][m] for x in
                                          sorted(algorithm_result.items())]
                        observations_b = [x[1][m] for x in
                                          sorted(algorithm_results[j].items())]
                        p_value = r_test(observations_a, observations_b)
                        f.write('\t%.2f' % p_value)
                    f.write('\n')
                f.write('\n')

    def results_overview(self, file_to_be_evaluated, output=sys.stdout,
                         scaletype='lin'):
        """ Method for getting a nice overview of results for a given file.

        This method is a helper method for generating the files

        - evaluation_overview_lin_scaled.txt
        - evaluation_overview_log_scaled.txt


        Parameters
        ----------
            file_to_be_evaluated (str): file name of the file containing
                the algorithm results which should be evaluated
            output (optional(str)): the output to which the results should
                be print
            scaletype (str): how to scale the scores ('lin' or 'log')

        Returns
        -------
            None

        """
        global queries
        algorithm_results = self.evaluate(file_to_be_evaluated,
                                          scaletype=scaletype)
        # get results sorted after appearance of queries
        # first print results for every query
        for query, results in sorted(algorithm_results.items(),
                                     key=lambda x: queries.index(x[0])):
            ndcg = results[0]
            acc_0 = results[1] * 100
            acc_1 = results[2] * 100
            acc_2 = results[3] * 100
            av = results[4]
            print("%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f" %
                  (query, ndcg, acc_0, acc_1, acc_2, av), file=output)
        print("\n", file=output)
        # now print average results
        a_ndcg = sum([x[0] for x in algorithm_results.values()]) / 10
        a_acc0 = sum([x[1] for x in algorithm_results.values()]) / 10 * 100
        a_acc1 = sum([x[2] for x in algorithm_results.values()]) / 10 * 100
        a_acc2 = sum([x[3] for x in algorithm_results.values()]) / 10 * 100
        a_av = sum([x[4] for x in algorithm_results.values()]) / 10
        print("# avr\t%.2f\t%.2f\t%.2f\t%.2f\t%.4f\n"
              % (a_ndcg, a_acc0, a_acc1, a_acc2, a_av), file=output)

    def score_distributions(self, files_to_be_evaluated, scaletype='lin'):
        """ Print score distributions for all given algorithms to a file.

        This method is used to generate the files

        - scores_scaled_lin.dat
        - scores_scaled_log.dat


        Parameters
        ----------
            files_to_be_evaluated (list[str]): filenames of the files
                containing the algorithm results for which the score
                distribution should be computed
            scaletype (str): how to scale the scores ('lin' or 'log')

        Returns
        -------
            None

        """
        with open('scores_scaled_%s.dat' % scaletype, 'w') as f:
            # print complete score lists of every algorithm and query in file
            for algorithm in files_to_be_evaluated:
                f.write('# %s\n' % algorithm)
                scores = self.get_ids_and_scores(algorithm)[1]
                if not algorithm == 'benchmark.txt':
                    scores = E.scale_scores(scores, scaletype=scaletype)
                # get the scores for every query
                queries = [str(x) for x in sorted(scores.keys())]
                f.write('# x %s\n' % ' '.join(queries))
                for i in range(149):
                    # print score
                    score_row = [scores[int(query)][i] for query in queries]
                    f.write('%d %s\n'
                            % (i+1,
                               ' '.join([str(score) for score in score_row])))
                # print score distribution
                score_counts = []
                for i in range(6):  # count all scores from 0 to 5
                    score_counts.append(
                        (i, sum([x.count(i) for x in scores.values()])
                         / 1490.0 * 100))
                f.write('# Score distribution of %s:\n# %s\n\n\n'
                        % (algorithm, score_counts))


def DCG(relevance_scores, k=14):
    """ Function for computing the discounted cumulative gain for a given
    list of relevance scores ordered by rank within result list.

    Examples
    --------
    >>> DCG([3, 2, 3, 0, 1, 2])
    8.097171433256849
    >>> DCG([3, 2, 3, 0, 1, 2], k=2)
    5.0

    """
    relevance_scores = relevance_scores[:k]
    DCG = relevance_scores[0]  # rel_1
    for i, relevance_score in enumerate(relevance_scores[1:]):
        position = i + 2  # position in relevance list
        DCG += relevance_score / math.log(position, 2)
    return DCG


def nDCG(relevance_scores, k=14):
    """ Function for computing the normalized discounted cumulative gain
    for a given list of relevance scores ordered by rank within result list.

    Examples
    --------
    >>> nDCG([3, 2, 3, 0, 1, 2])
    0.9315085232327253

    """
    # first compute DCG
    DCG_ = DCG(relevance_scores[:k])
    # compute DCG of ideal ordering
    IDCG = DCG(sorted(relevance_scores[:k], reverse=True))
    if DCG_ == 0:
        return 0
    return DCG_ / IDCG


def accuracy(returned_scores, actual_scores, deviation):
    """ Function for computing the accuracy for a given list of scores.

    The accuracy returns the percentage of scores in returned_scores
    that differ by at most deviation from the respective score in
    actual_scores.
    returned_scores is inteded to stem from an algorithm which should be
    evaluated and actual_scores are the scores from a given benchmark.

    Examples
    --------
    >>> accuracy([0, 2, 1, 3], [0, 0, 2, 3], 1)
    0.75
    >>> accuracy([0, 2, 1, 3], [0, 0, 2, 3], 2)
    1.0
    >>> accuracy([7, 7, 7, 7], [0, 0, 2, 3], 2)
    0.0

    """
    deviations = 0.0
    for i in range(len(returned_scores)):
        difference = abs(returned_scores[i] - actual_scores[i])
        if difference <= deviation:
            deviations += 1.0
    return deviations / len(returned_scores)


def scale_score(score, new_range=5.0, old_min=0, old_max=None,
                integer=False, reverse=False, scaletype='lin'):
    """ Function for scaling a given score to a given range.

    old_min and old_max are used to determine the old range.

    See http://stackoverflow.com/questions/929103/

    Examples
    --------
    >>> scale_score(0.5, new_range=4, old_min=0.0, old_max=1.0)
    2.0
    >>> scale_score(0.25, new_range=20, old_min=0.0, old_max=1.0)
    5.0
    >>> scale_score(0.3, new_range=3, old_min=0, old_max=1)
    0.8999999999999999
    >>> scale_score(0.3, new_range=3, old_min=0, old_max=1, integer=True)
    1.0
    >>> scale_score(0.25, new_range=20, old_min=0.0, old_max=1.0, reverse=True)
    15.0

    """
    old_range = old_max - old_min
    if (scaletype == 'lin'):
        new_score = (((score - old_min) * new_range) / old_range)
    elif (scaletype == 'log'):
        new_score = math.log((score / old_max) * 2 ** 5, 2)
    if reverse:
        new_score = new_range - new_score
    if integer and (scaletype == 'lin'):
        new_score = round(new_score)
    elif integer:  # rounding for logarithmic scaling
        if new_score >= 1:
            new_score = round(new_score)
        else:
            new_score = math.floor(new_score)
    return new_score


def eval_assignment(observations_a, observations_b, assignment):
    """ Compute the two means according to the given assignment as follows: if
    the i-th bit in the assignment is set to 0, the i-th observation will not
    be changed between observations_a and observations_b, if it is 1 the
    i-th observation will be changed between the two

    Examples
    --------
    >>> eval_assignment([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], int("000", 2))
    (2.0, 5.0)
    >>> eval_assignment([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], int("100", 2))
    (3.0, 4.0)
    >>> eval_assignment([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], int("111", 2))
    (5.0, 2.0)
    >>> eval_assignment([1.0], [2.0, 3.0], int("11", 2))
    Traceback (most recent call last):
     ...
    ValueError: The observation lists must have the same length
    """
    if not (len(observations_a) == len(observations_b)):
        raise ValueError("The observation lists must have the same length")
    n = len(observations_a)
    sum_a = 0
    sum_b = 0
    for i in range(n):
        if assignment & (1 << (n - i - 1)):  # i = 1, change position
            sum_a += observations_b[i]
            sum_b += observations_a[i]
        else:
            sum_b += observations_b[i]
            sum_a += observations_a[i]
    return (sum_a / n, sum_b / n)


def r_test(observation_a, observation_b):
    """ Compute the p-value according to the r-test for given obeservations.

    The two observations must have the same length.

    Examples
    --------
    >>> observation_a = [1, 3, 3, 5]
    >>> observation_b = [6, 6, 4, 4]
    >>> r_test(observation_a, observation_b)
    37.5
    >>> r_test(observation_a, observation_a)
    100.0

    """
    assignment_observed = int('0' * len(observation_a), 2)
    means_observed = eval_assignment(observation_a, observation_b,
                                     assignment_observed)
    delta_observed = abs(means_observed[0] - means_observed[1])
    n = len(observation_a)
    num_assignments = 1 << n
    count = 0
    # get delta for every possible combination
    for assignment in range(num_assignments):
        means = eval_assignment(observation_a, observation_b, assignment)
        delta = abs(means[0] - means[1])
        if delta >= delta_observed:
            count += 1.0
    return 100 * count / num_assignments


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    E = Evaluation('benchmark.txt')
    algorithms = ['resultlists_withoutmfcc.txt', 'resultlists_onlymfcc.txt',
                  'resultlists_allfeatures.txt', 'resultlists_freesound.txt']

    queries = [110011, 2155, 22362, 240015, 264498, 325462,
               337791, 50802, 5560, 82402]

    # get results for linear scaling
    with open('evaluation_overview_lin_scale.txt', 'w') as f:
        for algorithm in algorithms:
            f.write("# %s \n# query nDCG\tacc_0\tacc_1\tacc_2\tav\n"
                    % algorithm)
            E.results_overview(algorithm, output=f, scaletype='lin')

    # get results for logarithmic scaling
    with open('evaluation_overview_log_scale.txt', 'w') as f:
        for algorithm in algorithms:
            f.write("# %s \n# query nDCG\tacc_0\tacc_1\tacc_2\tav\n"
                    % algorithm)
            E.results_overview(algorithm, output=f, scaletype='log')

    # get significance for both scaling methods
    E.significance(algorithms, scaletype='lin')
    E.significance(algorithms, scaletype='log')

    # print score distributions of all algorithms and benchmark to file
    algorithms.append('benchmark.txt')
    E.score_distributions(algorithms, scaletype='lin')
    E.score_distributions(algorithms, scaletype='log')
