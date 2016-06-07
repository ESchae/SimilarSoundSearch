"""
Copyright 2016
Author: Elke Schaechtele <elke.schaechtele@web.de

This module is used for performing feature extraction on a given audiofile.
Essentia library must be installed to be able to use the feature extraction,
see http://essentia.upf.edu/ for the documentation and
https://github.com/MTG/essentia for the source code.

"""
import essentia  # needed explicitely for accessing logging module

# There were to different warnings encountered during implementation which
# both do not seem to affect the overall quality of the algorithm:
# 1: during loading mp3 files there is a 'Deprecated Audioconert' warning
# 2: for any import from essentia a 'ARGLLLLLL' warning
# Warnings are reported but not yet solved,
# see https://github.com/MTG/essentia/issues/410

# To not interrupt the main program with the two warnings essentia logging
# is disabled.
# Note: Should definitely be enabled again while further development.
essentia.log.warningActive = False


# needed algorithms from essentia
from essentia.standard import PoolAggregator
from essentia import Pool
from essentia.streaming import (MonoLoader,
                                Envelope,
                                RealAccumulator,
                                FrameCutter,
                                Windowing,
                                Spectrum,
                                DynamicComplexity,
                                Duration,
                                EffectiveDuration,
                                LogAttackTime,
                                Centroid,
                                PitchYinFFT,
                                MFCC)


class FeatureExtractor(essentia.streaming.CompositeBase):
    """ Class for feature extraction on a given audiofile.

    For feature extraction the streaming mode of Essentia is used. In streaming
    mode a network of connected algorithm is defined. Within this connection
    the data will automatically flow between algorithms.
    To perform feature extraction with this connection it is neough to call
    the method run on the algorithm from which the complete network is fed.

    All this will be done automatically during initialization of an instance
    of the FeatureExtractor class. This means, for performing feature
    extraction on an audiofile it is enough to instantiate a class instance
    with the filename of the audiofile.


    For more information see the documentation for the c++ streaming mode:
    http://essentia.upf.edu/documentation/streaming_architecture.html


    These features will be extracted:

    Global features
    - loudness
    - dynamicComplexity
    - logattacktime
    - duration
    - effectiveDuration

    Framebased features
    - pitch
    - pitchConfidence
    - spectralcentroid
    - mfcc (13 first coefficients)


    Attributes
    ----------
        _pool (pool): an essentia pool (dict-like object) which contains all
            extracted features with framebased features beeing not
            yet aggregated
        features (pool): the aggregated pool containing all global features
            and the aggregated framebased features
        feature_names (list[str]): the names of all extracted features
            if the default aggregation statistics are used these are:

            ['duration', 'dynamicComplexity', 'loudness', 'pitch.dmean',
            'pitch.dvar', 'pitch.mean', 'pitch.var', 'pitchConfidence.dmean',
            'pitchConfidence.dvar', 'pitchConfidence.mean',
            'pitchConfidence.var', 'spectralcentroid.dmean',
            'spectralcentroid.dvar', 'spectralcentroid.mean',
            'spectralcentroid.var', 'effectiveDuration',
            'logattacktime', 'mfcc.dmean', 'mfcc.dvar',
            'mfcc.mean', 'mfcc.var']

    """

    def __init__(self, filename, frameSize=2048, hopSize=1024,
                 window='hann', stats=['mean', 'var', 'dmean', 'dvar'],
                 sampleRate=44100):
        """ Initialize a feature extractor object for a given audiofile.

        By instantiation feature extraction will be automatically performed.
        The extracted features can be accessed via the attribute features
        or for the non aggreagted feature trajectories via the attribute pool.

        Parameters
        ----------
            filename (str): the filename of the audiofile for which features
                should be extracted
            frameSize (optional(int)): the size of the frames for the
                framebased features in samples, default=2048, note that the
                fast fourier transform is most efficient for a framesize
                which is a power of two
            hopSize (optional(int)): thw hop size between two consecutive
                frames, default=1024
            window (optional(str)): before computing the spectrum on a
                given frame it is necessary to window the signal with a given
                windowing function, possible options are: ['hamming', 'hann',
                'triangular', 'square', 'blackmanharris62', 'blackmanharris70',
                'blackmanharris74', 'blackmanharris92'], default='hann'
            stats (optional(list[str])): the statistics to be computed for the
                aggregation of framebased features, possible statistics are:
                ['min', 'max', 'median', 'mean', 'var', 'skew', 'kurt',
                'dmean', 'dvar', 'dmean2', 'dvar2'], with e.g.
                dmean and dmean2 being the first and second derivative of
                the mean, default=['mean', 'var', 'dmean', 'dvar']
            sampleRate (optional(int)): the desired output sampling rate,
                audiofiles with a different samplerate will be resampled


        Returns
        -------
            None


        Examples
        --------
        >>> audiofile = 'Testfiles/sine300.wav'
        >>> Extractor = FeatureExtractor(audiofile)
        >>> Extractor.features # doctest: +ELLIPSIS
        <essentia.common.Pool instance at 0x...>
        >>> Extractor.features['duration']
        0.30000001192092896
        >>> Extractor._pool['pitch'] # doctest: +NORMALIZE_WHITESPACE
        array([ 304.22268677,  301.05880737,  301.05871582,  301.05877686,
                301.05889893,  301.05886841,  301.05889893,  301.05880737,
                301.05880737,  301.05871582,  301.0586853 ,  301.05877686,
                301.05947876,  304.97198486], dtype=float32)
        >>> Extractor.features['pitch.mean']
        301.5643615722656
        >>> Extractor.features['pitchConfidence.mean']
        0.94275963306427

        """

        # instantiate as a feature extractor in streaming mode
        # this is done internal by essentia
        super(FeatureExtractor, self).__init__()

        # ------------------------------------------------------------------- #
        # -------- instantiate neccesary algorithms and connect them -------- #

        # ------------------------ preliminaries ---------------------------- #

        # the loader outputs the raw signal data from a given audiofile
        loader = MonoLoader(filename=filename, sampleRate=sampleRate)

        # pool where the feature values will be stored
        pool = Pool()

        # needed by logattacktime
        envelope = Envelope()
        accu = RealAccumulator()  # needed between logattacktime and envelope
        loader.audio >> envelope.signal
        envelope.signal >> accu.data

        # needed for framebased processing
        fc = FrameCutter(frameSize=frameSize, hopSize=hopSize)
        loader.audio >> fc.signal
        # windowing
        w = Windowing(type=window)
        fc.frame >> w.frame
        # spectrum
        spec = Spectrum()
        w.frame >> spec.frame

        # ------------------------- audio features -------------------------- #

        # ------------------------ global features -------------------------- #

        # dynamic complexity and loudness
        dynamicComplexity = DynamicComplexity()
        loader.audio >> dynamicComplexity.signal
        dynamicComplexity.dynamicComplexity >> (pool, 'dynamicComplexity')
        dynamicComplexity.loudness >> (pool, 'loudness')
        # duration
        duration = Duration()
        loader.audio >> duration.signal
        duration.duration >> (pool, 'duration')
        # effective duration
        effectiveDuration = EffectiveDuration()
        accu.array >> effectiveDuration.signal
        effectiveDuration.effectiveDuration >> (pool, 'effectiveDuration')
        # logattacktime
        log = LogAttackTime()
        accu.array >> log.signal
        log.logAttackTime >> (pool, 'logattacktime')

        # ---------------------- framebased features ------------------------ #

        # spectral centroid
        sc = Centroid()
        spec.spectrum >> sc.array
        sc.centroid >> (pool, 'spectralcentroid')
        # mfcc
        mfcc = MFCC(numberCoefficients=13)
        spec.spectrum >> mfcc.spectrum
        mfcc.bands >> None  # not included in feature vector
        mfcc.mfcc >> (pool, 'mfcc')
        # pitchYinFFT
        pitch = PitchYinFFT()
        spec.spectrum >> pitch.spectrum
        pitch.pitchConfidence >> (pool, 'pitchConfidence')
        pitch.pitch >> (pool, 'pitch')

        # ------------------ finished network connection -------------------- #
        # ------------------------------------------------------------------- #

        # start feature extraction
        essentia.run(loader)

        # aggregate results
        # logattacktime and effective duration are global features
        # but automatically aggregated in streaming mode
        # to handle this 'copy' is used
        aggrPool = PoolAggregator(
            defaultStats=stats, exceptions={'logattacktime': ['copy'],
                                            'effectiveDuration': ['copy']
                                            })(pool)
        self._pool = pool
        self.features = aggrPool
        self.feature_names = aggrPool.descriptorNames()

if __name__ == "__main__":
    import doctest
    doctest.testmod()
