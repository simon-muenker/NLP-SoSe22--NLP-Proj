# FrozenBERT+SentiDict

Warning: The following make calls are only available with OSX/LINUX. If you use Windows, please excuse the
inconvenience.

## Install

```bash
### to install the requirements (make, pip):
make install
pip install -r requirements.txt
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

### Config:

#### Base Config

```json5
{
  "cuda": 0,
  "out_path": "./", // path to results (log, model state, etc.)
  "data": {
    "paths": {
      "train": "./data/imdb.train.csv",
      "eval": "./data/imdb.eval.csv"
    },
    "polarities": {
      "negative": 0,
      "positive": 1
    },
    "data_label": "review",
    "target_label": "sentiment",
    "config": null // see data.default_config
  },
  "model": {
    // base class for neural classifier
    "neural": {
      "name": "Model Description"
    },
    // ... additional modules based on classifier type
    // text encoding (type: base, hybrid)
    "encoding": {
      "model": "bert-base-uncased"
    },
    // linguistic/statistical features (type: features, hybrid)
    "features": {
      "ngram_counter": {
        "1": 768,
        "2": 5120
      },
      "nela_pipeline": {
        "style": true,
        "complexity": true,
        "bias": true,
        "affect": true,
        "moral": false,
        "event": false
      }
    },
    // metacritic matcher (type: hybrid)
    "metacritic": {
      "path": "./data/metacritic.formatted.csv"
    },
  },
  "trainer": null // see trainer.default_config
}
```

## Results on eval (preliminary)

### Base

```bash
AVG           	 tp:     2023	 fp:      219 	 tn:     2023	 fn:      219	 pre=0.9023	 rec=0.9023	 f1=0.9023	 acc=0.9023
negative      	 tp:      925	 fp:      132 	 tn:     1098	 fn:       87	 pre=0.8751	 rec=0.9140	 f1=0.8942	 acc=0.9023
positive      	 tp:     1098	 fp:       87 	 tn:      925	 fn:      132	 pre=0.9266	 rec=0.8927	 f1=0.9093	 acc=0.9023
```

### Features

```bash
AVG           	 tp:     1947	 fp:      295 	 tn:     1947	 fn:      295	 pre=0.8684	 rec=0.8684	 f1=0.8684	 acc=0.8684
negative      	 tp:      862	 fp:      145 	 tn:     1085	 fn:      150	 pre=0.8560	 rec=0.8518	 f1=0.8539	 acc=0.8684
positive      	 tp:     1085	 fp:      150 	 tn:      862	 fn:      145	 pre=0.8785	 rec=0.8821	 f1=0.8803	 acc=0.8684
```

### Hybrid

```bash
AVG           	 tp:     2025	 fp:      217 	 tn:     2025	 fn:      217	 pre=0.9032	 rec=0.9032	 f1=0.9032	 acc=0.9032
negative      	 tp:      903	 fp:      108 	 tn:     1122	 fn:      109	 pre=0.8932	 rec=0.8923	 f1=0.8927	 acc=0.9032
positive      	 tp:     1122	 fp:      109 	 tn:      903	 fn:      108	 pre=0.9115	 rec=0.9122	 f1=0.9118	 acc=0.9032
```

## Credits:

* BERT Transformer by Huggingface: <https://huggingface.co/docs/transformers/model_doc/bert>
* NLP-Progress Scoreboard Sentiment Analysis (IMDb): <http://nlpprogress.com/english/sentiment_analysis.html#imdb>
