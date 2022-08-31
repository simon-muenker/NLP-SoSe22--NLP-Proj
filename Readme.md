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

```

## Credits:

* BERT Transformer by Huggingface: <https://huggingface.co/docs/transformers/model_doc/bert>
* NLP-Progress Scoreboard Sentiment Analysis (IMDb): <http://nlpprogress.com/english/sentiment_analysis.html#imdb>
