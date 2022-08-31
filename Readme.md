# Zero-Shot Classifier

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
# run test/debug (tiny dataset)
make debug

# run on full train with evaluation (not on test set)
make run
```

### Python module:

```bash
# Pass multiple json configs 
# CONFIGS := [ ./path/to/file.json, ... ] 
# python3 -m $classifier -C CONFIGS

# e.g.:
python3 -m $classifier -C config.json
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
  },
  "model": {
    // base class for neural classifier
    "neural": {
      "name": "Model Description"
    },
    // encoder model
    "encoding": {
      "model": "bert-base-uncased"
    }
  },
  "trainer": null // see trainer.default_config
}
```

## Results on eval (preliminary)

```
[--- EVAL -> ./data/imdb.eval.csv ---]
AVG           	 tp:     2021	 fp:      221 	 tn:     2021	 fn:      221	 pre=0.9014	 rec=0.9014	 f1=0.9014	 acc=0.9014
negative      	 tp:      938	 fp:      147 	 tn:     1083	 fn:       74	 pre=0.8645	 rec=0.9269	 f1=0.8946	 acc=0.9014
positive      	 tp:     1083	 fp:       74 	 tn:      938	 fn:      147	 pre=0.9360	 rec=0.8805	 f1=0.9074	 acc=0.9014
```

## Credits:

* BERT Transformer by Huggingface: <https://huggingface.co/docs/transformers/model_doc/bert>
* NLP-Progress Scoreboard Sentiment Analysis (IMDb): <http://nlpprogress.com/english/sentiment_analysis.html#imdb>
