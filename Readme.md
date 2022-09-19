# Zero-Shot Classifier

Warning: The following make calls are only available with OSX/LINUX. If you use Windows, please excuse the
inconvenience.

## Install

```bash
# installing the Python requirements
make install

# downloading the IMDb from its original source
# src: http://ai.stanford.edu/~amaas/data/sentiment/
make download

### run both
make setup
```

## Usage

```bash
# prepare datasets
make data

# analysis cluster distances and dispersion in different dimension using manifold reduction
make analysis

# train multi layered neural net to predict test data
make training

### run all sequentially
make run
```

## Results

### Encoder Analysis (Cluster analysis on full dimensional output)

#### Extra-Distance (train/test)

|                         |   mean |   std |    min |    max |
|:------------------------|-------:|------:|-------:|-------:|
| ('base', 'test')        |  1.293 | 0.179 |  1.125 |  1.603 |
| ('base', 'train')       |   1.36 | 0.206 |  1.193 |  1.775 |
| ('fabriceyhc', 'test')  | 19.999 | 0.085 | 19.964 | 20.239 |
| ('fabriceyhc', 'train') |  22.89 | 0.059 | 22.865 | 23.053 |
| ('textattack', 'test')  |  7.554 | 0.444 |  7.313 |  8.711 |
| ('textattack', 'train') |  8.053 | 0.424 |   7.82 |  9.156 |
| ('wakaka', 'test')      |  11.55 | 0.293 | 11.398 | 12.325 |
| ('wakaka', 'train')     | 11.629 | 0.286 | 11.483 | 12.393 |

#### Intra-Distance (train/test)

|                                     |  mean |   std |   min |   max |
|:------------------------------------|------:|------:|------:|------:|
| ('base', 'test', 'negative')        | 5.748 | 0.094 | 5.502 |   5.8 |
| ('base', 'test', 'positive')        | 6.047 |  0.06 | 5.906 | 6.087 |
| ('base', 'train', 'negative')       | 5.708 | 0.102 | 5.447 | 5.766 |
| ('base', 'train', 'positive')       | 6.101 | 0.059 | 5.966 |  6.14 |
| ('fabriceyhc', 'test', 'negative')  | 9.229 | 0.103 | 8.941 | 9.273 |
| ('fabriceyhc', 'test', 'positive')  | 7.376 | 0.141 | 6.985 | 7.438 |
| ('fabriceyhc', 'train', 'negative') | 7.596 | 0.103 | 7.314 | 7.644 |
| ('fabriceyhc', 'train', 'positive') |  5.98 | 0.142 | 5.584 | 6.041 |
| ('textattack', 'test', 'negative')  | 6.823 | 0.224 |  6.22 | 6.935 |
| ('textattack', 'test', 'positive')  |  7.06 | 0.246 | 6.399 | 7.184 |
| ('textattack', 'train', 'negative') | 6.618 | 0.236 | 5.987 | 6.738 |
| ('textattack', 'train', 'positive') | 6.998 |  0.24 | 6.351 | 7.118 |
| ('wakaka', 'test', 'negative')      | 7.481 | 0.203 | 6.933 | 7.581 |
| ('wakaka', 'test', 'positive')      | 8.336 | 0.221 | 7.746 | 8.447 |
| ('wakaka', 'train', 'negative')     | 7.335 | 0.209 | 6.768 | 7.437 |
| ('wakaka', 'train', 'positive')     | 8.459 | 0.207 | 7.901 | 8.562 |

### Classifier Training (F1/Loss density across epochs)

#### F1 (train/test)

![F1 Density on Train](results/training/f1.train.density.png?raw=true "F1 Density on Train")
![F1 Density on Test](results/training/f1.test.density.png?raw=true "F1 Density on Test")

#### Loss (train/test)

![Loss Density on Train](results/training/loss.train.density.png?raw=true "Loss Density on Train")
![Loss Density on Test](results/training/loss.test.density.png?raw=true "Loss Density on Test")

## Credits:

* BERT Transformer by Huggingface: <https://huggingface.co/docs/transformers/model_doc/bert>
* NLP-Progress Scoreboard Sentiment Analysis (IMDb): <http://nlpprogress.com/english/sentiment_analysis.html#imdb>
