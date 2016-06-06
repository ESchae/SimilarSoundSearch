"""
Copyright 2016
Author: Elke Schaechtele <elke.schaechtele@web.de>

This module contains code related to the work on CrowdFlower.
CrowdFlower was used for crouwdsourcing.
The crouwdsourcing results were used to establish the benchmark for evalution.

Basically the module contains a function crowdflower_csv to build the .csv file
for CrowdFlower and a class CrowdFlowerEvaluation which contains
utils for the evaluation of the crowdsourcing results.
"""
from __future__ import print_function
from __future__ import division
import json
import csv
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib

logging.basicConfig(level=logging.INFO)


def crowdflower_csv(jsonbase, query_ids, filename, sort=True):
    """ Helperfunction for building the crowdflower .csv data file.

    This function was used to build the .csv data file out of the
    json file containing the Freesound information on all sounds
    from D1. The function is inteded to be fed with a json file
    generated like the one that can be found in ../D1/audiofiles.json
    which can be build with the help of method querysets_json
    from class freesound_api in module freesound_utils.

    The function could be used for generating a random permutation
    of all rows, but the default is to get a list sorted first by
    query and then by cluster.

    The file will contain the following columns:

    - query_id : the id of one of the ten queries
    - result_id : the id of one of the 149 compared sounds from D1
    - 4 different preview types needed for displaying the sound
      in the crowdflower task

    Parameters
    ----------
        jsonbase (str): filename of json file from which the needed
            information can be picked
        query_ids (list[str]): the ids of the queries
        filename (str): name of the .csv file

    Returns
    -------
        None

    """
    with open(jsonbase) as f:
        db = json.loads(f.read())
    # query sounds will be contained in database as well
    all_ids = list(db.keys())
    rows = []
    headers = ['query_id', 'result_id', 'query-preview-lq-ogg',
               'query-preview-lq-mp3', 'result-preview-lq-ogg',
               'result-preview-lq-mp3']
    # get all combinations with query and every other sound
    for query in query_ids:
        logging.info('Combining %d with %d sounds in db'
                     % (query, len(db) - 1))
        for comparison in np.random.permutation(all_ids):
            # query will not be compared with itself
            if (int(comparison) == query):
                continue
            # add cluster id as prefix  to result_id if exists
            if db[comparison].get('target'):
                cluster = db[comparison]['target']
            else:
                cluster = "0"
            # each row has the form:
            # [query_id, result_id, query_ogg, query_mp3,  res_ogg, res_mp3]
            query_ogg = db[str(query)]['previews']['preview-lq-ogg']
            query_mp3 = db[str(query)]['previews']['preview-lq-mp3']
            res_ogg = db[str(comparison)]['previews']['preview-lq-ogg']
            res_mp3 = db[str(comparison)]['previews']['preview-lq-mp3']
            result_id = str(cluster) + "-" + str(comparison)
            # crowdflower need https instead of http
            rows.append([str(query),
                         result_id,
                         query_ogg.replace('http://', 'https://'),
                         query_mp3.replace('http://', 'https://'),
                         res_ogg.replace('http://', 'https://'),
                         res_mp3.replace('http://', 'https://')])
    logging.info("All combinations done. Num rows: %d " % len(rows))
    with open(filename, 'w') as f:
        print(", ".join(headers), file=f)
        if not sort:
            for row in np.random.permutation(rows):
                print(", ".join(row), file=f)
        else:
            # sort first by query_id and then by cluster
            rows.sort(key=lambda x: (int(x[0]), int(x[1].split('-')[0])))
            for row in rows:
                print(", ".join(row), file=f)


class CrowdFlowerEvaluation():
    """ Class for evaluating the results from CrowdFlower. """

    def __init__(self, crowdflower_aggr_all,
                 queries=[2155, 5560, 22362, 50802, 82402, 110011,
                          240015, 264498, 325462, 337791]):
        """ Evaluation must be initialised with an aggregated report of
        CrowdFlower results containing all answers per row.
        """
        self.aggr_all = crowdflower_aggr_all
        self.queries = queries

    def get_results(self):
        """ Generates a benchmark from the crowdflower results.

        The benchmark will be a dict containing the queries as keys.
        The values will be lists containing all compared sounds together with
        their crowdflower scores. The score is the number of judgments saying
        the two sounds are similar.

        The lists will be sorted by score in
        descending order.  -- nope! sorted by appearence in table!

        Examples
        --------
        >>> C = CrowdFlowerEvaluation('Complete/complete_aggr_all.csv')
        >>> results = C.get_results()
        >>> sorted(results.keys()) # doctest: +NORMALIZE_WHITESPACE
        [2155, 5560, 22362, 50802, 82402, 110011, 240015, 264498, \
        325462, 337791]
        >>> len(results[5560])
        149
        >>> results[5560] # doctest: +ELLIPSIS
        [(110011, 0), (2155, 0), (22362, 0), (240015, 0), ...]
        >>> results[110011][9:23]  # doctest: +NORMALIZE_WHITESPACE
        [(256452, 5), (62213, 5), (315829, 3), (62216, 3), (156643, 4), \
        (344872, 0), (62209, 4), (41251, 0), (169206, 0), (62215, 5), \
        (315834, 3), (339894, 0), (202192, 0), (133542, 1)]

        """
        # read the csv report
        with open(self.aggr_all) as f:
            reader = csv.DictReader(f)
            results = dict()
            # built the benchmark
            for row in reader:
                # get needed information from row
                # TODO: Warum manchmal 6 trusted judgments?
                answers = row['are_these_two_sounds_similar']
                score = answers.count('first_option')  # first_option = similar
                query = int(row['query_id'])
                result = row['result_id']
                result = int(result.split('-')[1])
                # append to benchmark
                if not results.get(query):
                    results[query] = []
                results[query].append((result, score))
            # sort results by appearence in table
            for query in results:
                result_list = results[query]
                results[query] = sorted(result_list,
                                        key=lambda x: all_sounds.index(x[0]))
        return results

    def get_cases(self, filename=None):
        """ Method for retrieving the four cases:

        1) dissimilar sounds in same cluster (not expected)
        2) dissimilar sounds in different cluster (expected)
        3) similar different cluster (not expected)
        4) similar same cluster (expected)

        The expectations stem from the naive assumptions that the 10 clusters
        retrieved by freesounds similarity search contain only similar sounds
        and there are not similarities between the clusters.

        Parameters
        ----------
            filename (optional(str)): if specified result will be print to file

        Returns
        -------
            cases_dict (dict[str][int]): a dict containing nested dicts
                first there are for dicts (one for every case) and then there
                are dicts for every query within this case

        Examples
        --------
        >>> C = CrowdFlowerEvaluation('Complete/complete_aggr_all.csv')
        >>> cases = C.get_cases()
        >>> dissimilars = cases['dissimilar_same_cluster'][5560]
        >>> dissimilars  # doctest: +NORMALIZE_WHITESPACE
        ['5560-321026(0)', '5560-143951(2)', '5560-156845(0)', \
        '5560-61967(0)', '5560-61680(0)', '5560-255209(1)', '5560-81192(1)']
        >>> x = cases['similar_different_cluster'][5560]
        >>> x # doctest: +NORMALIZE_WHITESPACE
        ['5560-17786(3)(2155)', '5560-61680(4)(337791)', \
         '5560-143951(3)(337791)']

        """
        cases = ['dissimilar_same_cluster', 'dissimilar_different_cluster',
                 'similar_different_cluster', 'similar_same_cluster']
        # built dict from the given cases
        cases_dict = dict()
        for case in cases:
            cases_dict[case] = dict()
            # get cases for every query
            for cluster in self.queries:
                cases_dict[case][cluster] = []
        # seperate dict for checking similarity within query sounds
        cases_dict['similar_queries'] = []
        with open(self.aggr_all) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # get needed information from the csv row
                answers = row['are_these_two_sounds_similar']
                score = answers.count('first_option')  # first_option = similar
                query = int(row['query_id'])
                prefix, result = [int(x) for x in row['result_id'].split('-')]
                similar = True if (score >= 3) else False
                if prefix != 0:
                    cluster = prefix
                else:  # in this row two queries were compared
                    if score != 0:  # they are not completely dissimilar
                        info = '%d-%d(%d)' % (query, result, score)
                        cases_dict['similar_queries'].append(info)
                    continue
                # insert into belonging case
                info_same = '%d-%d(%d)' % (prefix, result, score)
                info = '%d-%d(%d)(%d)' % (prefix, result, score, query)
                if not similar and cluster == query:
                    cases_dict[
                        'dissimilar_same_cluster'][cluster].append(info_same)
                elif not similar and cluster != query:
                    cases_dict[
                        'dissimilar_different_cluster'][cluster].append(info)
                elif similar and cluster != query:
                    cases_dict[
                        'similar_different_cluster'][cluster].append(info)
                elif similar and cluster == query:
                    cases_dict[
                        'similar_same_cluster'][cluster].append(info_same)
        if filename:
            with open(filename, 'w') as f:
                json.dump(cases_dict, f, indent=4, separators=(',', ':'))
        return cases_dict

    def similarity_matrix(self):
        """ Method for computing a similarity matrix.

        A row in the matrix belongs to a query and the columns are all sounds
        from D1 in their appearence as specified in the table.

        Returns
        -------
            similarity_matrix : the computed matrix
            queries : the ten queries used for computing the matrix
            all_sounds : the global variable all_sounds


        Examples
        --------
        >>> C = CrowdFlowerEvaluation('Complete/complete_aggr_all.csv')
        >>> matrix, queries, all_sounds = C.similarity_matrix()
        >>> matrix.shape
        (10, 150)

        """
        queries = all_sounds[:10]
        similarity_matrix = np.zeros([10, 150], dtype=int)
        results = self.get_results()
        for i, query in enumerate(queries):
            result_list = results[query]
            # add query to resultlist
            result_list.insert(i, (query, 5))
            # now only keep the scores
            scores = [x[1] for x in result_list]
            # and append every row to the matrix
            similarity_matrix[i] = scores
        return similarity_matrix, queries, all_sounds

    def plot_similarity_matrix(self, filename=None):
        """ Method for plotting the similarity matrix nicely.

        With the matrix the results from CrowdFlower can be displayed clearly.

        Parameters
        ----------
            filename (optionale(str)): if a filename is given the plot will be
                saved to the file, otherwise it will be displayed directly

        Returns
        -------
            None

        """
        matplotlib.rcParams['text.usetex'] = True
        matrix, queries, all_sounds = self.similarity_matrix()
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        cax = ax.matshow(matrix, cmap='YlOrRd', vmin=0, vmax=5,
                         extent=[0, 150, 0, 10], aspect=4)
        cbar = fig.colorbar(cax, boundaries=[0, 1, 2, 3, 4, 5, 6],
                            shrink=0.5, aspect=30, pad=0.01)
        cbar.set_label(label=r'\"Ahnlichkeitsgrad', size=8.4)
        cbar.set_ticks([x for x in range(6)])
        cbar.ax.set_yticklabels([0, 1, 2, 3, 4, 5], va='bottom', size=9)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        ax.set_xticks([x for x in range(10, 150, 14)])
        ax.set_xticks([x for x in range(150)], minor=True)
        ax.set_yticks([x for x in range(10)])

        labels = [r'Katze', r'Hydrant', r'T\"ur', r'Spielzeug', r'Uhr',
                  r'Lachen', r'Regen', r'Grollen', r'Feuerwerk', r'Schrei']

        # ax.set_xticklabels(queries, rotation=0, ha='left')
        ax.set_xticklabels(['Cluster %d\n%s' % (i + 1, x)
                            for (i, x) in enumerate(labels)],
                           ha='left', rotation=10)
        ax.set_xticklabels(['Bspkl.\n1-10'], minor=True, rotation=10,
                           ha='left')
        ax.set_yticklabels(['%s - %d' % (x, 10 - i)
                            for (i, x) in enumerate(reversed(labels))],
                           va='bottom')
        # major ticks
        ax.tick_params(axis='both', which='major', labelsize=8.4,
                       direction='out', length=3, pad=1, bottom='off',
                       right='off')
        ax.tick_params(axis='x', which='major', pad=-3, length=18)
        # minor ticks
        ax.tick_params(axis='x', which='minor', labelsize=8.4,
                       direction='out', length=5, bottom='off', right='off')
        ax.grid(which='minor', alpha=0.2, linestyle='-')
        ax.grid(which='major', alpha=1.0, linestyle='-')
        if not filename:
            plt.show()
        else:
            plt.savefig(filename, dpi=800, bbox_inches='tight')

    def score_distribution(self):
        """ Method for computing the distribution og the similarity scores.

        Returns
        -------
            score_distribution (list): a list containing tuples of
                (score, percentage)


        Examples
        --------
        >>> logging.basicConfig(level=logging.ERROR)
        >>> C = CrowdFlowerEvaluation('Complete/complete_aggr_all.csv')
        >>> C.score_distribution()
        [(0, 80.0), (1, 9.0), (2, 4.0), (3, 2.0), (4, 2.0), (5, 3.0)]

        """
        with open(self.aggr_all) as f:
            reader = csv.DictReader(f)
            scores = []
            for row in reader:
                answers = row['are_these_two_sounds_similar']
                num_answers = len(answers.split('\n'))
                if num_answers != 5:
                    logging.warning('%d answers for task id %s, deleting last'
                                    % (num_answers, row['id']))
                    answers = answers.split('\n')[:5]
                score = answers.count('first_option')
                scores.append(score)
            score_distribution = []
            for i in range(6):
                percentage = round(100 * (scores.count(i) / len(scores)), 0)
                score_distribution.append((i, percentage))
            return score_distribution

    def benchmark_to_file(self, filename):
        """ Method for generating a file benchmark.txt for evaluation.

        Every line will start with a query followed by a tab.
        After the tab there follows a sorted list of space separated
        id=score pairs for every sound compared to the query.
        The list is sorted by score (see function benchmark).

        Parameters
        ----------
            filename (str) : file where to save the benchmark

        Returns
        -------
            None

        """
        benchmark = self.get_results()
        # sort all result lists by score in descending order
        for query in benchmark:
            benchmark[query].sort(key=lambda x: x[1], reverse=True)
        with open(filename, 'w') as f:
            for query, result_list in benchmark.items():
                query_row = '%d\t' % int(query)
                for result, score in result_list:
                    query_row += '%d=%d ' % (result, score)
                query_row += '\n'
                f.write(query_row)


# a list of all sound ids, starting with the query sounds
# after this for each query there follow the 14 similar sounds from freesound
# ordered by decreasing similarity score
# needed for matrix plotting
# note: this is the same order as the order of the table in the appendix
# of the corresponding Bachelor's thesis
all_sounds = [110011, 2155, 22362, 240015, 264498, 325462, 337791, 50802,
              5560, 82402, 256452, 62213, 315829, 62216, 156643, 344872,
              62209, 41251, 169206, 62215, 315834, 339894, 202192, 133542,
              140391, 165116, 135003, 190936, 327542, 329045, 167206, 210827,
              326445, 321444, 70626, 201635, 202529, 241318, 174067, 247658,
              248927, 249016, 247842, 250953, 249015, 248926, 69552, 247459,
              248609, 247456, 247731, 247880, 266319, 334805, 124716, 118560,
              278141, 118561, 181679, 296486, 245193, 216994, 223676, 254755,
              70377, 210178, 125319, 174604, 174252, 240280, 174353, 185045,
              264456, 174251, 186134, 30608, 174148, 179385, 248045, 344633,
              344039, 251489, 40728, 37241, 89467, 200837, 119450, 316793,
              319346, 4237, 55209, 7285, 19158, 45129, 213012, 326441, 79367,
              197202, 110612, 130231, 191876, 80834, 44695, 44734, 209650,
              93985, 163615, 276928, 66574, 167277, 156091, 170989, 343282,
              74408, 178925, 240734, 71153, 116831, 163445, 83930, 83946,
              84521, 239135, 30303, 207322, 17786, 199944, 255209, 81192,
              156845, 321026, 189446, 61680, 214052, 143951, 61967, 326332,
              242009, 248875, 66550, 69504, 302793, 248660, 175395, 246989,
              321284, 339820, 42909, 66545, 82418]


if __name__ == '__main__':
    import doctest
    doctest.testmod()
