# Zero-Shot Classifier

Warning: The following make calls are only available with OSX/LINUX. If you use Windows, please excuse the
inconvenience.

## Install

```bash
# to install the requirements (make, pip):
make install

# to download the raw IMDb dataset
make download
```

## Usage

### Predefined Targets:

```bash
# prepare datasets (originals must be downloaded manually)
make data

# analysis cluster distances and dispersion in different dimension using manifold reduction
make analysis

# train multi layered neural net to predict test data
make training
```

## Credits:

* BERT Transformer by Huggingface: <https://huggingface.co/docs/transformers/model_doc/bert>
* NLP-Progress Scoreboard Sentiment Analysis (IMDb): <http://nlpprogress.com/english/sentiment_analysis.html#imdb>
