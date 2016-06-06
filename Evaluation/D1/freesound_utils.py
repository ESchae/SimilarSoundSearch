"""
Copyright 2016
Author: Elke Schaechtele <elke.schaechtele@web.de>

This is a module intended to be used as client to Freesound's API
see http://www.freesound.org/docs/api/

Note: You need your own client secret to work with the API.
You can request one at http://www.freesound.org/apiv2.

The doctest will not work until you specify your own client secret.
Note that testing the module will take some seconds as there are
tested some requests which need some time.

"""
from __future__ import print_function
import requests
import os.path
import logging
import time
import json
import csv

logging.basicConfig(level=logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# your client secret from Freesound's API
client_secret = None


class freesound_api():
    """ Class for retrieving needed sound data from Freesound's API.

    For working with the API you first need to request your own
    API credential at http://www.freesound.org/apiv2
    The secret key you will be given there can then be used for
    authentication within this class.
    """

    def __init__(self, key=client_secret):
        """ Instantiates the authorization header for every api request.

        Parameters
        ----------
            key (str): client secret of Freesound's API

        >>> a = freesound_api('my_client_secret')
        >>> a.headers
        {'Authorization': 'Token my_client_secret'}

        """
        # generate header which is needed in every request for authentication
        self.headers = {'Authorization': 'Token ' + key}

    def sound_instance(self, sound_id,
                       fields=['id', 'url', 'name', 'duration', 'samplerate',
                               'previews', 'similar_sounds'],
                       flat=False):
        """ Method for requesting sound instances by id.

        The sound instance is a dictionary including the specified fields.
        If needed, nested fields can be flattened, e.g.
        from 'previews' - 'previews-lg-mp3', 'previews-hq-mp3', ... into
        'previews$preview-lq-mp3', 'previews$preview-hq-mp3', ...
        For more information on the sound instances see
        https://www.freesound.org/docs/api/resources_apiv2.html#sound-instance

        Parameters
        ----------
            sound_id (int): the id of the sound to be retireved
            fields (optional(list[str])): fields for narrow the request,
                for alternatives to the defaut fields see:
                default fields:
                    id (int): the sound id
                    url (str): url to the sound
                    name (str): name given by the user who uploaded the sound
                    duration (float): duration of the sound in seconds
                    samplerate (float): samplerate of the sound
                    previews (dict[str:str]): urls to four different types
                        of previews (high/low quality and .mp3 or .ogg files):
                        'preview-lq-ogg', 'preview-lq-mp3',
                        'preview-hq-ogg', 'preview-hq-mp3'
                    similar_sounds (str): url to page with results of similar
                        sounds search
            flat (optional(bool)): whether or not to flatten nested fields
                like 'previews'

        Returns
        -------
            instance (dict): dict object with the given fields as keys


        Examples
        --------
        >>> a = freesound_api()
        >>> sound = a.sound_instance(123, flat=True)
        >>> sound['id']
        123
        >>> sound['name']
        u'guitar_phone.wav'
        >>> sound['previews$preview-lq-ogg']
        u'http://www.freesound.org/data/previews/0/123_23-lq.ogg'
        >>> type(sound)
        <type 'dict'>

        """
        # the request url
        url = "http://www.freesound.org/apiv2/sounds/" + str(sound_id) + "/"
        # only return specified fields (or all fields if none are given)
        if fields:
            url += "?fields=" + ",".join(fields)
        logging.info("\tRequesting sound instance with id %d..." % sound_id)
        # start request
        request = requests.get(url, headers=self.headers)
        # check for errors with connection
        if request.status_code != 200:
            logging.info("Ups! Status code %d" % request.status_code)
            logging.info("Url was %s" % url)
            logging.info(request.text)
        instance = request.json()  # decode json format
        if flat:
            instance = flatten(instance)
        return instance

    def sound_instances_to_dict(self, sound_ids,
                                fields=["id", "url", "name", "duration",
                                        "samplerate", "previews",
                                        "similar_sounds"],
                                flat=False,
                                duration_limit=None,
                                min_samplerate=None):
        """ Method for saving several sound instanced into one dict.

        Builds a nested dict containing all instanced of the form

        {sound_id1 : {field1 : x, field2 : y, ...}, {sound_id2 : {...}, ...}

        If needed, the inner dicts can be flattened (see description of
        sound instance).
        Per default only sounds with duration <= 20 sec and a
        samplerate >= 44100 will be included.

        Parameters
        ----------
            sound_ids (list[ints]): a list of all sound ids
            fields (optional(list[str])): see description of sound instance
            flat (optional(bool)): whether or not to flatten nested fields
            duration_limit (optional(int)): include only sounds with a
                duration <= duration_limit (in seconds), default=20
            min_samplerate (optional(int)): include only sounds with a
                samplerate >= min_samplerate, default=44100

        Returns
        -------
            instances (dict[sound_id:sound_instance]): a nested dict
                containing all the sound instanced of the given sound ids


        Examples
        --------
        >>> a = freesound_api()
        >>> d = a.sound_instances_to_dict([123, 5560])
        >>> sorted(d.keys())
        [123, 5560]
        >>> sorted(d[123].keys()) # doctest: +NORMALIZE_WHITESPACE
        [u'duration', u'id', u'name', u'previews', u'samplerate',
        u'similar_sounds', u'url']

        """
        # build dict for all sound instances
        instances = dict()
        # count variables for logging
        l = len(sound_ids)
        inserted = 0
        checked = 0
        logging.info("\tBuilding dict containig soundinstances %s" % sound_ids)
        # insert every sound instance
        for sound_id in sound_ids:
            time.sleep(1)  # to prevent throttling
            sound_instance = self.sound_instance(sound_id, fields=fields,
                                                 flat=flat)
            logging.info("\t[%d/%d] Checking sound instance with id %d"
                         % (checked, l, sound_id, ))
            include = self.check_including(sound_instance)
            checked += 1
            if include:
                instances[sound_id] = sound_instance
                logging.info("\t... was included")
                inserted += 1
        logging.info("\tFinished: %d of %d included" % (inserted, l))
        return instances

    def similar_sound_instances_to_dict(self, sound_id,
                                        fields=['id', 'url', 'name',
                                                'duration', 'samplerate',
                                                'previews'],
                                        flat=False,
                                        duration_limit=None,
                                        min_samplerate=None,
                                        max_similars=None):
        """ Method for building a dict of similar sound instances.

        Builds a nested dict containing all similar sound instanced to
        a given sound id of the form

        {similar_sound_id1 : {field1 : x, field2 : y, ...},
        {similar_sound_id2 : {...}, ...}

        The inner dicts contain the given fields plus distance_to_target
        and target per default, where target is the given sound id wo which
        the sound are similar.
        The given sound_instance itself will not be included.
        Per default only sounds with duration <= 20 sec and a
        samplerate >= 44100 will be included.

        Parameters
        ----------
            sound_id (int): the sound id to which similar sounds should be
                retrieved
            fields (optional(list[str])): see description of sound instance
            flat (optional(bool)): whether or not to flatten nested fields
            duration_limit (optional(int)): include only sounds with a
                duration <= duration_limit (in seconds), default=20
            min_samplerate (optional(int)): include only sounds with a
                samplerate >= min_samplerate, default=44100
            max_similars (optional(int)): maximum of similar sounds to be
                returned, if None all similar sounds which do not violate one
                of the above constraints will be included, maximum returned
                by freesound api are 14 similar sounds

        Returns
        -------
            similars (dict[similar_id:similar_sound]): a dict containing
                all the similar sound instances


        Examples
        --------
        >>> a = freesound_api()
        >>> similar = a.similar_sound_instances_to_dict(1234, \
max_similars=2, duration_limit=23)
        >>> sorted(similar.keys())
        [7474, 7478]
        >>> sorted(similar[7474].keys()) # doctest: +NORMALIZE_WHITESPACE
        ['distance_to_target', u'duration', u'id', u'name', u'previews',
        u'samplerate', 'target', u'url']

        """
        logging.info("Requesting similar sounds for id %d" % sound_id)
        # get a list of all similar sounds
        similar_sounds = self.similar_ids_and_distance(sound_id)
        similars = dict()
        # count variables for logging
        l = len(similar_sounds)
        inserted = 0
        checked = 0
        # insert all similar sound instanced in similars dict
        logging.info("Building dict containing similar sound instances %s"
                     % [s[0] for s in similar_sounds])
        for similar_id, distance in similar_sounds:
            # get similar sound instance per id
            time.sleep(1)  # to prevent throttling
            sound_instance = self.sound_instance(similar_id, fields=fields)
            # add target and distance_to_target field
            sound_instance['target'] = sound_id
            sound_instance['distance_to_target'] = distance
            logging.info("\t[%d/%d]Checking similar sound instance %d"
                         % (checked, l, similar_id))
            include = self.check_including(sound_instance,
                                           duration_limit=duration_limit,
                                           min_samplerate=min_samplerate)
            checked += 1
            if include:
                similars[similar_id] = sound_instance
                logging.info("\t... was included")
                inserted += 1
            if len(similars) == max_similars:
                logging.info("\t%d sounds included - MAX SIMILARS reached"
                             % max_similars)
                return similars
        logging.info("\tFinished: %d of %d included" % (inserted, l))
        return similars

    def check_including(self, sound_instance, duration_limit=None,
                        min_samplerate=None):
        """
        Examples
        --------
        >>> a = freesound_api()
        >>> sound_instance = a.sound_instance(123)
        >>> sound_instance['duration']
        6.78458049887
        >>> sound_instance['samplerate']
        44100.0
        >>> a.check_including(sound_instance, duration_limit=5) is None
        True
        >>> a.check_including(sound_instance, duration_limit=30,
        ...                   min_samplerate=44200) is None
        True
        """
        sound_id = sound_instance['id']
        if duration_limit and sound_instance['duration'] > duration_limit:
            logging.info("\t... %s was excluded because of duration %d"
                         % (sound_id, sound_instance['duration']))
            return None
        elif min_samplerate and sound_instance['samplerate'] < min_samplerate:
            logging.info("\t... %s was excluded because of samplerate %d"
                         % (sound_id, sound_instance['samplerate']))
            return None
        else:
            return True

    def similar_ids_and_distance(self, sound_id, max_sounds=20):
        """ Returns a list of tuples containing
        similar sound id and their distance to target to a given sound id.

        >>> a = freesound_api()
        >>> a.similar_ids_and_distance(1234, max_sounds=3)
        [(7474, 0.7881477475166321), (7475, 0.8859057426452637)]

        """
        # request url for retrieving similar sounds
        url = "http://www.freesound.org/apiv2/sounds/" + str(sound_id)
        url += "/similar/"
        # similar sounds (dicts)
        similar_sounds = requests.get(url,
                                      headers=self.headers).json()['results']
        similar_sounds = similar_sounds[1:]  # first sound is sound itself
        # only ids and distance to target needed
        return [(s['id'], s['distance_to_target'])
                for s in similar_sounds][:max_sounds-1]

    def download(self, url, filename, path=None):
        """ Download the sound from a given url to specified path.

        Parameters
        ----------
            url (str): url from which the sound can be downloaded
            filename (str): filename for the sound
            path (str): path where to save the soundfile, if None file will
                be saved at current working directory

        Returns
        -------
            None

        """
        response = requests.get(url, headers=self.headers)
        abs_path = os.path.join(path, filename)
        logging.info("Downloading from %s to %s" % (url, abs_path))
        with open(abs_path, "wb") as f:
            f.write(response.content)

    def download_similar_sounds(self, sound_id, path=None, duration_limit=None,
                                min_samplerate=None, max_similars=None,
                                preview_type='preview-lq-mp3'):
        """ Method for downloading similar sounds to a given sound id.

        Per default the low quality mp3 version will be downloaded.
        Each sound in the directory will be named in the following manner:

        distance_to_target _ id _ . fileending

        Parameters
        ----------
            sound_id (int): id of the sound to which retrieve similar sounds
            path (optional(str)): path where to download the soundfiles,
                if no path is specified everything will be saved to cwd
            duration_limit (optional(int)): include only sounds with a
                duration <= duration_limit (in seconds), default=20
            min_samplerate (optional(int)): include only sounds with a
                samplerate >= min_samplerate, default=44100
            max_similars (optional(int)): maximum of similar sounds to be
                returned, if None all similar sounds which do not violate one
                of the above constraints will be included, maximum returned
                by freesound api are 14 similar sounds
            preview_type (optional(str)): which preview type should be
                downloaded, options: 'preview-lq-ogg', 'preview-hq-mp3',
                'preview-hq-ogg', 'preview-lq-mp3', default='preview-lq-mp3'


        Returns
        -------
            None

        """
        # first download sound itself
        sound = self.sound_instance(sound_id)
        filename = "0-" + str(sound_id) + "." + preview_type.split("-")[-1]
        url = sound['previews'][preview_type]
        self.download(url, filename=filename, path=path)
        # get a dict containing all similar sounds to the sound
        similar_sounds = self.similar_sound_instances_to_dict(
            sound_id, duration_limit=duration_limit,
            min_samplerate=min_samplerate, max_similars=max_similars)
        # download all the sound
        for similar_id, similar_sound in similar_sounds.items():
            url = similar_sound['previews'][preview_type]
            # filename: sound_id - distance - similar_id . file_ending
            distance = "%.2f" % similar_sound['distance_to_target']
            filename = "-".join([str(sound_id), distance, str(similar_id)])
            filename += "." + preview_type.split("-")[-1]  # file ending
            self.download(url,  filename=filename, path=path)

    def querysets_json(self, start_ids, filename,
                       fields=["id", "url", "name", "duration", "samplerate",
                               "previews", "similar_sounds"], flat=False,
                       duration_limit=None, min_samplerate=None,
                       max_similars=None):
        """ Generates a json file containing information on all sounds given
        in start_ids and their similar sounds.

        This method was used for building the audiofiles.json file.
        For every sound information will be stored in the following manner:

        <Freesound_id>: {
                "name": <name on Freesound>,
                "url": <url to Freesound>,
                "previews": {
                    "preview-lq-ogg": <url to low-quality .ogg preview>,
                    "preview-lq-mp3": <url to low-quality .mp3 preview>,
                    "preview-hq-ogg": <url to high-quality .ogg preview>,
                    "preview-hq-mp3": <url to high-quality .mp3 preview>
                },
                "distance_to_target": <distance to the start_id (target)>,
                "duration": <duration in seconds>,
                "samplerate": <samplerate>,
                "similar_sounds": <url to similar sounds>,
                "id": <the id again>,
                "target": <id of the start_id (target)>
        }

        Parameters
        ----------
            start_ids (list[int]): the ids for which to include similar sounds
            filename (str): filename where to save the generated json file
            fields (optional(list[str])): fields for narrow the request,
                for alternatives to the defaut fields see:
                default fields:
                    id (int): the sound id
                    url (str): url to the sound
                    name (str): name given by the user who uploaded the sound
                    duration (float): duration of the sound in seconds
                    samplerate (float): samplerate of the sound
                    previews (dict[str:str]): urls to four different types
                        of previews (high/low quality and .mp3 or .ogg files):
                        'preview-lq-ogg', 'preview-lq-mp3',
                        'preview-hq-ogg', 'preview-hq-mp3'
                    similar_sounds (str): url to page with results of similar
                        sounds search
            duration_limit (optional(int)): include only sounds with a
                duration <= duration_limit (in seconds), default=20
            min_samplerate (optional(int)): include only sounds with a
                samplerate >= min_samplerate, default=44100
            max_similars (optional(int)): maximum of similar sounds to be
                returned, if None all similar sounds which do not violate one
                of the above constraints will be included, maximum returned
                by freesound api are 14 similar sounds

        Returns
        -------
            None

        """
        logging.info("Building json database at '%s'" % filename)
        j = dict()
        # add start sound instances to dict
        logging.info("Adding %d start_ids" % len(start_ids))
        instances = self.sound_instances_to_dict(start_ids, fields=fields,
                                                 flat=flat,
                                                 duration_limit=duration_limit)
        j.update(instances)
        # add similar sound instances to dict
        logging.info("Adding similar sounds to start_ids")
        for start_id in start_ids:
            similars = self.similar_sound_instances_to_dict(
                start_id, fields=fields, flat=flat,
                duration_limit=duration_limit, max_similars=max_similars)
            j.update(similars)
        logging.info("FINISHED building database with %d sounds" % len(j))
        logging.info("Writing json formatted to file %s" % filename)
        # write json like dict to file
        with open(filename, "w") as f:
            json.dump(j, f, indent=4)

    def distances(self, query_id, sounds_to_be_compared):
        """ Returns the distance between one sound and any number of sounds.

        Combined search is needed for this, therefore the request might take
        some time. If you use a lot of sounds this might really really take
        some time.

        Note: The usage of the Freesound API is limited to max 2000 requests
        per day. If you need to compare a lot of sounds you might want to ask
        the administrators to give you more permissive limits.

        Parameters
        ----------
            query_id (int): the id of the sound
            sounds_to_be_compared (list[int]): all sounds for which the
                distance the the given query should be returned
            output_file (str): file where results will be print to

        Returns
        -------
            distances (list[(str, str, int)]): a list of tuples containing
                id, distance of this id to the query and page at which the
                result was found during combined search

        Examples
        --------
        >>> a = freesound_api()

        """
        # prepare string containing sounds for reuqest url
        filter_sounds = '+OR+'.join(['id:%d' % sound
                                     for sound in sounds_to_be_compared])
        url = 'http://www.freesound.org/apiv2/search/combined/?'
        url += 'target=%s&filter=(%s)' % (query_id, filter_sounds)
        # during combined search is not guaranteed that results will be
        # at the first page, therefore you need to iterate all pages
        # until every needed distance is found
        # this is why we need the following counters
        pages = 1
        found_distances = 0
        distances = []
        # start request
        logging.info('Get distance between %d and %s'
                     % (query_id,  sounds_to_be_compared))
        while True:
            try:
                request = requests.get(url, headers=self.headers)
                result = request.json()
                if not result['results']:  # go on searching on next page
                    logging.info('No results on page %d...' % pages)
                    pages += 1
                    url = result['more']
                else:
                    # there may be more than one result at this page
                    for i, s in enumerate(result['results']):
                        sound = result['results'][i]['id']
                        distance = result['results'][i]['distance_to_target']
                        logging.info('Found distance %s at page %d to sound %s'
                                     % (str(distance), pages, sound))
                        distances.append((sound, distance, pages))
                        found_distances += 1
                    if (found_distances == len(sounds_to_be_compared)):
                        # all distances found
                        break
                    else:
                        url = result['more']
                        pages += 1
            except KeyError:
                logging.info(result)  # some problem qith request

    def all_distances(self, all_sounds, queries,
                      output_file='freesound_distances.txt'):
        """ Method for getting all distances between sounds and another
        set of sounds.

        Results will be printed to a given file.
        Thie method was used for retrieving all Freesound intern distances
        between the 10 queries and all 150 sounds within D1.

        It is assumed that queries are << than sounds_to_be_compared.

        The file will contain the following columns:
        - query id (from one of the sounds)
        - sound id (from one of the sounds to be compared)
        - distance (between the two according to Freesound)
        - page (in which page during combined search the distance was found)

        Parameters
        ----------
            all_sounds(list[int]): the sounds which sould be compared with the
                queries
            queries (list[int]): the queries to which all sounds should be
                compared

        Returns
        -------
            None

        """
        with open(output_file, 'w') as f:
            for sound in all_sounds:
                distances = self.distances(sound, queries)
                for query, distance, pages in distances:
                    f.write('%s, %d, %s, %d\n' %
                            (query, sound, distance, pages))

    def freesound_original_top_150_distances(self, queries, output_file):
        """ Method for retrieving the distances for the first 150 similar
        sounds to every of the given query.

        This method was used to get the original Freesound distances for the
        ten queries from D1. Not that the first 14 distances will belong
        to sounds from D1, but the following may be any sounds from Freesound.

        The file will contain one column for every query and 149 rows
        in which the distances for every similar sound and query is written.

        Parameters
        ----------
            queries (list[int]): a list of the queries for which the distances
                of the top 150 similar should be retrieved
            output_file (str): file where the results should be print to

        Returns
        ------
            None

        """
        with open(output_file, 'w') as f:
            all_distances = []
            for query in queries:
                logging.info('Getting 150 most similar sounds to %d' % query)
                url = 'http://www.freesound.org/apiv2/sounds/'
                url += '%d/similar/?page_size=150' % query
                request = requests.get(url, headers=self.headers)
                result = request.json()['results']
                distances = [str(round(x['distance_to_target'], 2))
                             for x in result][1:]  # exclude query itself
                all_distances.append(distances)
            # print distances to file
            for i in range(149):
                f.write("%d %s\n" %
                        (i + 1, ' '.join([x[i] for x in all_distances])))

    def freesound_results(self, input_file='freesound_distances.txt',
                          output_file='results_freesound.txt'):
        """ Method for generating the needed file for evaluation.

        The file will have the same form like the benchmark, this means
        for every query of D1 it contains a line. A line starts with the query
        followed by a tab and then the corresponding resut list as space
        sparated id=scores pairs.

        The input file was generated with the help of method all_distances
        and sorted then first after query (first column) and then with
        increasing distance (third column).

        Parameters
        ----------
            input_file (optional(str)): filename of the input file (s. above)
            output_file (optional(str)): filename of the output file (s. above)

        Returns
        -------
            None

        """
        # built result dictionary with query as keys
        results = dict()
        with open(input_file) as f:
            reader = csv.reader(f)
            for row in reader:
                query, sound_id, distance = row[:3]
                sound_id = sound_id.strip()
                if (query == sound_id):  # do not include query in result list
                    continue
                if not results.get(query):
                    results[query] = []
                results[query].append((sound_id,
                                       str(round(float(distance), 3))))
        # write to benchmark-like file
        with open(output_file, 'w') as f:
            for query, result_list in sorted(results.items()):
                results = ' '.join(['='.join(x) for x in result_list])
                f.write('%s\t%s\n' % (query, results))


def flatten(d, parent_key='', sep='$'):
    """ Flattens a nested dict.
    Source: http://stackoverflow.com/questions/6027558/

    >>> d = {'id':1, 'pre':{'ogg':'x', 'mp3':'y'}}
    >>> sorted(flatten(d).items())
    [('id', 1), ('pre$mp3', 'y'), ('pre$ogg', 'x')]

    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
