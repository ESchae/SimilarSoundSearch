# About this folder

This folder contains all information about the dataset 1 (D1) used for evaluation
of the implemented algorithm. This means basically there is the complete dataset
along with complementary information as well as the code used to get the files
via [Freesound's API](http://www.freesound.org/docs/api/).

## Contents

* /audiofiles --> the .mp3 files of all 150 sounds from D1 (better suited for processing or reproducing results or get more information related to freesound about the files)

* /audiofiles_ better_ to_read --> the .mp3 files from D1 (better suited for human reading)

* audiofiles.csv --> overview of the 150 files with short descriptions in German (this is exact the same table as can be found in the appendix of the corresponding Bachelor's thesis)

* audiofiles.json --> contains Freesound related information on every sound in the form
```javascript
<Freesound id>: {
        "name": <name on Freesound>, 
        "url": <url to Freesound>, 
        "previews": {
            "preview-lq-ogg": <url to the low quality preview in .ogg file format>, 
            "preview-lq-mp3": <url to the low quality preview in .mp3 file format>, 
            "preview-hq-ogg": <url to the high quality preview in .ogg file format>, 
            "preview-hq-mp3": <url to the high quality preview in .mp3 file format>
        }, 
        "distance_to_target": <distance to the sound given in target field>, 
        "duration": <duration in seconds>, 
        "samplerate": <samplerate>, 
        "similar_sounds": <url to similar sounds>, 
        "id": <the id again>, 
        "target": <id of the target sound>
}
```

* freesound_distances.txt --> contains the Freesound distances from all ten queries of D1 to every of the 150 sounds from D1
* freesound_origina _distances.txt --> contains for every of the ten queries from D1 the distances to the top 150 search results of Freesound's similarity search
* freesound_utils.py --> contains the code used for working with Freesound's API
* queries.mp3 --> the ten queries in the short version in one file as used as example in the thesis
