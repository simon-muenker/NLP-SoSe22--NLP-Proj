# FrozenBERT+SentiDict
Warning: The following make calls are only available with OSX/LINUX. If you use Windows, please excuse the inconvenience.

## Install
```bash
### to install and download the spacy/textblob pipelines use:
make install

### manually install and download:
pip install -r requirements.txt

python -m spacy download en_core_web_sm
python -m textblob.download_corpora
```

## Usage

### Predefined Targets:
```bash
# run test/debug (tiny datasets)
make debug

# run all experiments
make ex

# run individual sub experiments:
# TYPE := [linguistic, transformer, hybrid]
# SIZE := [1.000, 0.100, 0.010]
# make ex_[TYPE]_train.[SIZE]

# e.g.:
make ex_linguistic_train.0.010
```
### Python module:
```bash
# Target the individual modules and pass multiple json configs 
# CONFIGS := [ ./path/to/file.json, ... ] 
# python3 -m $classifier.[TYPE] -C CONFIGS

# e.g.:
python3 -m $classifier.linguistic -C ./global.json ./model.json
```

#### Config:
TODO

## Results (TBC)
TODO

## Credits:

* NLP-Progress Sentiment Analysis (IMDb): <http://nlpprogress.com/english/sentiment_analysis.html#imdb>
