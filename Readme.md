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
# TYPE := [base, features, hybrid]
# make ex_[TYPE]

# e.g.:
make ex_features
```
### Python module:
```bash
# Target the individual modules and pass multiple json configs 
# CONFIGS := [ ./path/to/file.json, ... ] 
# python3 -m $classifier.[TYPE] -C CONFIGS

# e.g.:
python3 -m $classifier.base -C ./global.json ./model.json
```

#### Config:
ToDo

## Results (preliminary)

### Base
```bash
AVG           	 tp:     1826	 fp:      655 	 tn:     1826	 fn:      655	 pre=0.7360	 rec=0.7360	 f1=0.7360	 acc=0.7360
negative      	 tp:      891	 fp:      345 	 tn:      935	 fn:      310	 pre=0.7209	 rec=0.7419	 f1=0.7312	 acc=0.7360
positive      	 tp:      935	 fp:      310 	 tn:      891	 fn:      345	 pre=0.7510	 rec=0.7305	 f1=0.7406	 acc=0.7360
```

### Features
```bash
AVG           	 tp:     2081	 fp:      400 	 tn:     2081	 fn:      400	 pre=0.8388	 rec=0.8388	 f1=0.8388	 acc=0.8388
negative      	 tp:      946	 fp:      145 	 tn:     1135	 fn:      255	 pre=0.8671	 rec=0.7877	 f1=0.8255	 acc=0.8388
positive      	 tp:     1135	 fp:      255 	 tn:      946	 fn:      145	 pre=0.8165	 rec=0.8867	 f1=0.8502	 acc=0.8388
```

### Hybrid
```bash
AVG           	 tp:     2138	 fp:      343 	 tn:     2138	 fn:      343	 pre=0.8617	 rec=0.8617	 f1=0.8617	 acc=0.8617
negative      	 tp:     1040	 fp:      182 	 tn:     1098	 fn:      161	 pre=0.8511	 rec=0.8659	 f1=0.8584	 acc=0.8617
positive      	 tp:     1098	 fp:      161 	 tn:     1040	 fn:      182	 pre=0.8721	 rec=0.8578	 f1=0.8649	 acc=0.8617
```

## Credits:

* BERT Transformer by Huggingface: <https://huggingface.co/docs/transformers/model_doc/bert>
* English NLP Pipeline (small) by Spacy: <https://spacy.io/models/en#en_core_web_sm>
* Polarity, Subjectivity Pipeline by Textblob: <https://spacytextblob.netlify.app>
* NLP-Progress Scoreboard Sentiment Analysis (IMDb): <http://nlpprogress.com/english/sentiment_analysis.html#imdb>
