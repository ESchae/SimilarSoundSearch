# About this folder

This folder contains all information about the dataset 1 (D1) used for evaluation
of the implemented algorithm. This means basically there is the complete dataset
along with complementary information as well as the code used to get the files
via [Freesound's API](http://www.freesound.org/docs/api/).

## Contents

* /audiofiles --> the .mp3 files of all 150 sounds from D1
* audiofiles.csv --> overview of the 150 files with short descriptions in German (this is exact the same table as can be found in the appendix of the corresponding Bachelor's thesis)
* audiofiles.json --> contains information on every sound in the form
<<<<<<< HEAD


=======
* ```json
>>>>>>> 3cda41796c4f0e0565e0a7227d3792a09f9e0f26
"<Freesound_id>": {
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
