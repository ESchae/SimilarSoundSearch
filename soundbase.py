"""
Copyright 2016 University of Freiburg
Elke Schaechtele <elke.schaechtele@web.de>

This module contains the functionality associated with building a sql database
(aka. soundbase) from given audiofiles containing information on
path and audiofeatures of each of the audiofiles per row.
The audiofeatures will be extracted with the default settings specified
in FeatureExtractor class.

For working with the sql database the python library dataset is used, see
http://dataset.readthedocs.io/en/latest/api.html

"""
# for information on the following logging disabling
# see module feature_extraction
import essentia
essentia.log.warningActive = False

import dataset
import logging
import os
import pickle
import glob
from feature_extractor import FeatureExtractor
from tqdm import tqdm
from multiprocessing import Process, Pipe
import time

logger = logging.getLogger('soundbase')
logging.basicConfig()
logging.getLogger('alembic').setLevel(logging.WARNING)


class SoundBase(object):
    """ Class for working with a sql database containing sounds.

    A filled SoundBase consists of a table 'sounds' with a row for each
    sound and the columns id, path as well as one for every extracted feature.
    Features will be extracted with the FeatureExtractor class and their
    values will be serialized. To speed up queries all columns will be
    indexed per default.

    Note: During insertion it will be not checked if a sound already exists
    in the soundbase. Automatic updating is not yet supported, hence you need
    to check yourself that you do not add the same sound multiple times.

    Attributes
    ----------
        filename (str): filename
        db (database): direct connection to the database
        sounds (table): direct connection to the table in the database

    """

    def __init__(self, filename, delete_existing=False):
        """ Instantiates connection to a database with given filename.

        If the database already exists it can be overwritten, if desired.

        Parameters
        ----------
            filename (str): desired filename of the database
            delete_existing (optional(boolean)): whether or not a existing
               database with the same filename should be deleted
               and written new, default=False

        Returns
        -------
            None


        Examples
        --------
        >>> db = SoundBase('Testfiles/test.db', delete_existing=True)
        >>> os.path.isfile('Testfiles/test.db')
        True
        >>> db.db
        <Database(sqlite:///Testfiles/test.db)>
        >>> db.sounds
        <Table(sounds)>

        """
        self.filename = filename
        # handle case that database already exists
        if os.path.isfile(self.filename):
            logger.info('SoundBase %s already exists' % self.filename)
            if delete_existing:
                logger.info('Deleting existing SoundBase %s' % self.filename)
                os.remove(self.filename)
        logger.info("Connecting with SoundBase %s" % self.filename)
        self.db = dataset.connect('sqlite:///%s' % self.filename)  # database
        self.sounds = self.db['sounds']  # table

    def list_features_in_database(self):
        """ Method for retrieving the column names of the extracted features
        in the database.

        Returns
        -------
            list[str]: names of columns of the extracted features


        Examples
        --------
        >>> db = SoundBase('Testfiles/test.db', delete_existing=True)
        >>> db.list_features_in_database() is None
        True
        >>> db = db.fill('Testfiles', disable_progressbar=True)
        >>> db.list_features_in_database()  # doctest: +NORMALIZE_WHITESPACE
        ['pitchConfidence_dvar', 'mfcc_mean', 'spectralcentroid_dvar',
        'duration', 'mfcc_dvar', 'pitchConfidence_var', 'logattacktime',
        'pitch_dmean', 'dynamicComplexity', 'spectralcentroid_dmean',
        'spectralcentroid_var', 'pitchConfidence_mean', 'pitch_dvar',
        'pitch_mean', 'mfcc_dmean', 'pitch_var', 'mfcc_var',
        'pitchConfidence_dmean', 'spectralcentroid_mean', 'effectiveDuration',
        'loudness']


        """
        cols = self.sounds.columns
        try:
            cols.remove('id')
            cols.remove('path')
            return cols
        except ValueError:  # no features were extracted yet
            return None

    def num_rows(self):
        """ Method for directly retrieving the number of rows in the database.

        Returns
        -------
            num_rows (int): number of rows (sounds) in the database


        Examples
        --------
        >>> db = SoundBase('Testfiles/test.db', delete_existing=True)
        >>> db.num_rows()
        0
        >>> db = db.fill('Testfiles', disable_progressbar=True)
        >>> db.num_rows()
        5

        """
        return self.sounds.count()

    def fill(self, directory, ext=['wav', 'mp3', 'ogg'], recurse=True,
             limit=None, file_output=None, disable_progressbar=False):
        """ Fills the SoundBase with all audiofiles found at directory.

        Parameters
        ----------
            directory (str): path to directory where audiofiles are located
            ext (optional(list[str]): file extensions which should be
                included, default=['mp3', 'wav', 'ogg'], additional supported
                types: ['flac']
            recurse (optional(boolean)): whether or not all subfolders of
                directory shhould be searched, default=True
            limit (optional(int>0 or None)): at most limit files are returned,
                if None all files are returned
            file_output (optional(boolean)): if a filename is given the
                database wille freezed as json file with given name
            stats (optional(list[str])): statistics to be computed for
                aggregation of framebased features, default=['mean', 'var'],
                see method extract_features of class Sound for more details
            disable_progressbar (optional(bool)): whether or not to display
                a progressbar in the output during insertion, default='False'

        Returns
        -------
            self


        Examples
        --------
        >>> db = SoundBase('Testfiles/test.db', delete_existing=True)
        >>> f = 'Testfiles'
        >>> db = db.fill(f, disable_progressbar=True)
        >>> for row in db.sounds: print(row['id'], row['path'])
        (1, u'Testfiles/sine300.wav')
        (2, u'Testfiles/sine440.wav')
        (3, u'Testfiles/sine880.mp3')
        (4, u'Testfiles/test_recurse/test.wav')
        (5, u'Testfiles/test_short.wav')

        """
        # check if directory exists
        if not os.path.isdir(directory):
            raise IOError('There is no directory %s ' % directory)

        # get all the audiofiles
        audiofiles = find_files(directory, ext=ext, recurse=recurse,
                                limit=limit)

        # check if directory contains any audiofile
        num_audiofiles = len(audiofiles)
        if num_audiofiles == 0:
            raise IOError('No valid audiofiles found at directory')

        # add the rows for all audiofiles to the database
        logger.info('Filling SoundBase %s with %d audiofiles (100 at a time)'
                    % (self.filename, num_audiofiles))
        # get features for audiofiles in packages of 100 audiofiles at a time
        # This is needed because of an internal memory leak of Essentia.
        # To circumvent the memory leak multiprocessing is used, see
        # https://github.com/MTG/essentia/issues/54 second to last entry
        # The issue is not fixed yet, as can be seen here:
        # https://github.com/MTG/essentia/issues/405
        average_time = 0
        packages = 0
        for i in tqdm(range(0, num_audiofiles, 100),  # tqdm is the progressbar
                      disable=disable_progressbar):
            # get package (100 files)
            try:
                package = audiofiles[i:i+100]
            except IndexError:  # only less than 100 left
                package = audiofiles[i:]

            start_time = time.time()
            # initiate a parent process and a child process which
            # are connected by a pipe
            parent_conn, child_conn = Pipe()
            # initialize the childprocess
            p = Process(
                # target is the function executing the feature extraction
                # the childprocesses will work on this function
                target=self._collect_rows,
                # the arguments given to function _collect_rows
                args=(package,
                      child_conn,  # for connection between parent and child
                      disable_progressbar))
            p.start()  # start the childprocess
            rows = parent_conn.recv()  # receive message from childprocess

            # add all 100 rows of the package to the database
            # it is faster to add rows in packages than adding them one by one
            # see http://dataset.readthedocs.io/en/latest/api.html
            self.sounds.insert_many(rows)
            p.join()  # join the process
            end_time = time.time()
            average_time += end_time - start_time
            packages += 1
        logger.info('\t...added files in %.2f s on average per 100'
                    % (average_time / packages))

        # create index in database for every column
        self.sounds.create_index(self.sounds.columns)

        if file_output:
            # generate json file from actual database
            result = self.db['sounds'].all()
            dataset.freeze(result, format='json', filename=file_output)
        return self

    def _collect_rows(self, audiofiles, conn=None, disable_progressbar=False):
        """ Helperfunction for method fill to insert all sounds at once.

        Returns a list of dicts. Each dict specifies the row values for
        one audiofile whereby the keys are the columns in the SoundBase.
        This is done to be able to use the inser_many function from library
        dataset, as it is said to be significantly faster than adding rows
        one by one, see http://dataset.readthedocs.io/en/latest/api.html

        Parameters
        ----------
            audiofiles (list[str]): list of all audiofiles to be included
            conn (optional(_multiprocessing.Connection)): the connection to
                the parent process in method fill (see documentations
                there for details)
            disable_progressbar (optional(bool)): whether or not to display
                a progressbar in the output during insertion, default='False'

        Returns
        -------
            rows (list[dicts]): the row specifications for each audiofile


        Examples
        --------
        >>> audiofiles = ['Testfiles/sine300.wav', 'Testfiles/test_short.wav']
        >>> db = SoundBase('Testfiles/test.db', delete_existing=True)
        >>> db = db.fill('Testfiles', disable_progressbar=True)
        >>> rows = db._collect_rows(audiofiles, disable_progressbar=True)
        >>> len(rows)
        2
        >>> rows[0]['path']
        'Testfiles/sine300.wav'
        >>> import pickle
        >>> pickle.loads(rows[0]['spectralcentroid_mean'])
        0.015583268366754055

        """
        rows = []
        # generate data dict for each audiofile and append it to rows
        for audiofile in tqdm(audiofiles, disable=disable_progressbar):
            # add features
            features = FeatureExtractor(audiofile).features
            row = _pool_to_dict(features)
            # add relative path
            row['path'] = os.path.relpath(audiofile)
            # id column will be created per default during insertion
            # in database
            rows.append(row)
        if conn:
            conn.send(rows)  # send rows to parent process in method fill
            conn.close()
        return rows

    def get_row(self, row_id):
        """ Method for simplified access to a database row per id.

        Parameters
        ----------
            row_id (int): id of row belonging to a sound

        Returns
        -------
            row (dict): a dict containing the column names as keys and
                corresponding values of the sound with given id


        Examples
        --------
        >>> db = SoundBase('Testfiles/test.db', delete_existing=True)
        >>> db = db.fill('Testfiles', disable_progressbar=True)
        >>> row = db.get_row(1)
        >>> row['id']
        1
        >>> row['path']
        u'Testfiles/sine300.wav'

        """
        return self.sounds.find(id=row_id).__next__()

    def get_id(self, filename):
        """ Method for retrieving the id of an entry via a given filename.

        Parameters
        ----------
            filename (str): the filename for which the id should be returned

        Returns
        -------
            if (int): the corresponding id to the filename if there exists
                an entry with this filename in the database,
                if not an exception will be thrown

        Examples
        --------
        >>> db = SoundBase('Testfiles/test.db', delete_existing=True)
        >>> db = db.fill('Testfiles', disable_progressbar=True)
        >>> db.get_id('Testfiles/sine300.wav')
        1
        >>> db.get_id('doesnotexist.wav')
        Traceback (most recent call last):
         ...
        IOError: No sound with path doesnotexist.wav in the soundbase

        """
        try:
            return self.sounds.find(path=filename).__next__()['id']
        except StopIteration:
            raise IOError('No sound with path %s in the soundbase' % filename)


def _pool_to_dict(pool, serialize=True):
    """ Helpermethod to convert a pool object into a python dict.

    A pool is a data structure from Essentia used in FeatureExtractor class.
    Although a pool is already a dict-like object a python dict
    is necessary for writing the sound object to a database like
    specified in class SoundBase. This might be optimized in the future.

    Parameters
    ----------
        pool (pool): the pool which should be converted to a dictionary
        serialize (optional(bool)): whether or not the entries of the pool
            should be serialized or not, default=True

    Returns
    -------
        d (dict): the converted pool as dict


    Examples
    --------
    >>> from essentia.standard import PoolAggregator
    >>> from essentia import Pool
    >>> pool = Pool()
    >>> pool.add('a', 1)
    >>> pool.add('a', 2)
    >>> pool = PoolAggregator(defaultStats=['mean'])(pool)
    >>> pool.add('b', 3)
    >>> d = _pool_to_dict(pool, serialize=False)
    >>> sorted(d.items())
    [('a_mean', 1.5), ('b', array([ 3.], dtype=float32))]
    >>> d_serialized = _pool_to_dict(pool, serialize=True)
    >>> sorted(d_serialized.items()) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    [('a_mean', 'F1.5\\n.'),
    ('b', "cnumpy.core.multiarray\\n_reconstruct...\\ntp14\\nb.")]

    """
    if serialize:
        d = {key.replace('.', '_'): pickle.dumps(pool[key])
             for key in pool.descriptorNames()}
    else:
        d = {key.replace('.', '_'): pool[key]
             for key in pool.descriptorNames()}
    return d


def find_files(directory, ext=['wav', 'mp3', 'ogg', 'flac'],
               recurse=True, limit=None):
    """ Get a sorted list of audiofiles from a directory.

    A slim version of function find_files from librosa library; see
    https://github.com/bmcfee/librosa/ in librosa/util/files.py

    Function was almost copied here because it is useful, but it would
    not be useful to import librosa just because of one function as it first
    most be installed completely with all its dependencies.

    Parameters
    ----------
        directory (str): path to directory where files are located
        ext (optional(list[str]): file extensions which should be
            included, supportet types: flac, mp3, ogg, wav,
            default=['mp3', 'wav', 'ogg', 'flac']
        recurse (optional(boolean)): whether or not all subfolders of
            directory shhould be searched, default=True
        limit (optional(int>0 or None)): at most limit files are returned,
            if None all files are returned

    Returns
    -------
        audiofiles (list[str]): a sorted list of the audiofiles in directory

    Examples
    --------
    >>> print(find_files('Testfiles', ext=['wav'])) # doctest: +ELLIPSIS
    ['...sine300.wav', '...sine440.wav', '...test.wav', '...test_short.wav']
    >>> print(find_files('Testfiles', recurse=False)) # doctest: +ELLIPSIS
    ['...sine300.wav', '...sine440.wav', '...sine880.mp3', '...test_short.wav']
    >>> print(find_files('Testfiles', limit=2)) # doctest: +ELLIPSIS
    ['...sine300.wav', '...sine440.wav']
    >>> find_files('aksj') is None
    True
    >>> find_files('') is None
    True
    >>> find_files(None) is None
    True

    """
    # check border cases
    if directory is None:
        return None
    elif not os.path.isdir(directory):
        return None
    # collect files
    files = []
    if recurse:
        for walk in os.walk(directory):
            files.extend(__get_files(walk[0], ext=ext))
    else:
        files = __get_files(directory, ext=ext)
    files.sort()
    if limit is not None:
        files = files[:limit]
    return files


def __get_files(directory, ext=['wav', 'mp3', 'ogg', 'flac']):
    """ Helper function to get files in a single directory.

    This function is completely taken from librosa library; see
    https://github.com/bmcfee/librosa/ in librosa/util/files.py
    """
    # expand out the directory
    dir_name = os.path.abspath(os.path.expanduser(directory))
    myfiles = []
    for sub_ext in ext:
        globstr = os.path.join(dir_name, '*' + os.path.extsep + sub_ext)
        myfiles.extend(glob.glob(globstr))
    return myfiles


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    logging.getLogger('soundbase').setLevel(logging.INFO)

    # console arguments handling
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate a sql database (aka. soundbase) containing \
path and audiofeatures for all audiofiles found at a given directory.')
    parser.add_argument('filename', type=str,
                        help='filename of the soundbase')
    parser.add_argument('directory', type=str,
                        help='directory to look for audiofiles')
    parser.add_argument('-d', '--delete',
                        help='delete existing soundbase with given filename, \
default=False', action='store_true')
    parser.add_argument('-o', type=str,
                        help='freeze database to given filename')
    parser.add_argument('-ext', '--extensions', type=str,
                        help='which file extension to include, \
possible options: mp3, flac, ogg, wav, default=mp3,flac,ogg (must be specified\
comma separated without blank inbetween')
    parser.add_argument('-r', '--recurse', action='store_true',
                        help='search directory recursive for audiofiles')
    parser.add_argument('-l', '--limit', type=int,
                        help='only add limit audiofiles')
    parser.add_argument('-npb', '--noprogressbar', action='store_true',
                        help='do not show the progressbar')

    # passed arguments
    args = parser.parse_args()

    # get file extensions
    available_exts = ['flac', 'mp3', 'ogg', 'wav']
    if args.extensions:
        exts = args.extensions.split(',')
        for ext in exts:
            if ext not in available_exts:
                raise IOError('fileformat %s not supported,\
use one of mp3, ogg, flac, wav' % ext)
    else:  # use default exts
        exts = ['wav', 'mp3', 'ogg']

    s = SoundBase(args.filename, delete_existing=args.delete)
    start_time = time.time()
    s.fill(args.directory, ext=exts, recurse=args.recurse, limit=args.limit,
           file_output=args.output, disable_progressbar=args.noprogressbar)
    end_time = time.time()
    logger.info('Finished after %.2f s' % (end_time - start_time))
