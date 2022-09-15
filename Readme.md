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

## Cluster Analysis

dimension: [768, 576, 384, 192, 96, 48, 24, 12, 6, 3]

### Distance

#### Train

|                            |          mean |      std |    min |    max |
|:---------------------------|--------------:|---------:|-------:|-------:|
| ('base', 'distance')       |         1.388 |    0.224 |  1.194 |  1.853 |
| ('textattack', 'distance') |         8.053 |    0.422 |  7.82  |  9.145 |
| ('fabriceyhc', 'distance') |    **22.889** | **0.06** | 22.865 | 23.058 |
| ('wakaka', 'distance')     |        11.629 |    0.284 | 11.483 | 12.383 |


#### Test

|                            |     mean |        std |    min |    max |
|:---------------------------|---------:|-----------:|-------:|-------:|
| ('base', 'distance')       |     1.23 |      0.151 |   0.95 |  1.508 |
| ('textattack', 'distance') |    7.553 |      0.443 |  7.313 |  8.709 |
| ('fabriceyhc', 'distance') |   **20** |  **0.087** | 19.964 | 20.243 |
| ('wakaka', 'distance')     |   11.555 |      0.303 | 11.398 | 12.355 |

### Dispersion

#### Train

|                                       |        mean |     std |         min |         max |
|:--------------------------------------|------------:|--------:|------------:|------------:|
| ('base', 'dispersion_positive')       | 1.62761e+06 | 27256.9 | 1.56168e+06 | 1.64453e+06 |
| ('base', 'dispersion_negative')       | 1.57115e+06 | 37430.4 | 1.47652e+06 | 1.59272e+06 |
| ('textattack', 'dispersion_positive') | 1.87574e+06 | 76759.1 | 1.67287e+06 | 1.91578e+06 |
| ('textattack', 'dispersion_negative') | 1.82716e+06 | 76015.5 | 1.62743e+06 | 1.86737e+06 |
| ('fabriceyhc', 'dispersion_positive') | 1.60961e+06 | 42549.9 | 1.49247e+06 | 1.62892e+06 |
| ('fabriceyhc', 'dispersion_negative') | 2.04075e+06 | 34859.7 | 1.94572e+06 | 2.05708e+06 |
| ('wakaka', 'dispersion_positive')     | 2.26326e+06 | 65963.3 | 2.08894e+06 | 2.29731e+06 |
| ('wakaka', 'dispersion_negative')     | 2.02705e+06 | 70413.9 | 1.8407e+06  | 2.0633e+06  |


#### Test

|                                       |        mean |     std |         min |         max |
|:--------------------------------------|------------:|--------:|------------:|------------:|
| ('base', 'dispersion_positive')       | 1.49514e+06 | 23230.1 | 1.43447e+06 | 1.50908e+06 |
| ('base', 'dispersion_negative')       | 1.68364e+06 | 35953.3 | 1.58858e+06 | 1.70359e+06 |
| ('textattack', 'dispersion_positive') | 1.74989e+06 | 71602.2 | 1.56039e+06 | 1.78715e+06 |
| ('textattack', 'dispersion_negative') | 2.0028e+06  | 77080.5 | 1.79946e+06 | 2.04319e+06 |
| ('fabriceyhc', 'dispersion_positive') | 1.81558e+06 | 39690.7 | 1.70687e+06 | 1.8339e+06  |
| ('fabriceyhc', 'dispersion_negative') | 2.59127e+06 | 39343.7 | 2.48365e+06 | 2.60941e+06 |
| ('wakaka', 'dispersion_positive')     | 2.05835e+06 | 64845.2 | 1.88756e+06 | 2.09203e+06 |
| ('wakaka', 'dispersion_negative')     | 2.19356e+06 | 74640.7 | 1.99588e+06 | 2.23199e+06 |


## Classifier Training

### Optimization Process

|   epoch | ('base', 'f1_train') |   ('base', 'f1_test') |   ('textattack', 'f1_train') |   ('textattack', 'f1_test') |   ('fabriceyhc', 'f1_train') |   ('fabriceyhc', 'f1_test') |   ('wakaka', 'f1_train') |   ('wakaka', 'f1_test') |
|--------:|---------------------:|----------------------:|-----------------------------:|----------------------------:|-----------------------------:|----------------------------:|-------------------------:|------------------------:|
|      10 |             0.892985 |              0.887505 |                     0.978517 |                    0.934156 |                     0.987471 |                    0.931333 |                 0.905192 |                0.902101 |
|      20 |             0.902662 |              0.88948  |                     0.984259 |                    0.93218  |                     0.988034 |                    0.931212 |                 0.915271 |                0.901536 |
|      30 |             0.912501 |              0.891738 |                     0.986949 |                    0.930809 |                     0.988756 |                    0.931696 |                 0.923021 |                0.903189 |
|      40 |             0.921375 |              0.887827 |                     0.990202 |                    0.930285 |                     0.990041 |                    0.930245 |                 0.931454 |                0.904883 |
|      50 |              0.93029 |              0.884642 |                     0.991447 |                    0.931293 |                     0.990764 |                    0.92968  |                 0.937397 |                0.898875 |
|      60 |             0.932378 |              0.890851 |                     0.992732 |                    0.9296   |                     0.991969 |                    0.930164 |                 0.945509 |                0.900165 |
|      70 |             0.938281 |              0.886617 |                     0.993736 |                    0.930325 |                     0.992491 |                    0.929398 |                 0.947195 |                0.897827 |
|      80 |             0.942055 |              0.891174 |                     0.994137 |                    0.933108 |                     0.992611 |                    0.928632 |                 0.95105  |                0.902625 |
|      90 |             0.944906 |              0.882384 |                     0.994097 |                    0.930083 |                     0.993133 |                    0.928551 |                 0.953741 |                0.900085 |
|     100 |             0.950367 |              0.882384 |                     0.995342 |                    0.930285 |                     0.993455 |                    0.931495 |                 0.95631  |                0.903673 |

### Testset evaluation
| label    |   ('base', 'prec') |   ('base', 'rec') |   ('base', 'f1') |   ('textattack', 'prec') |   ('textattack', 'rec') |   ('textattack', 'f1') |   ('fabriceyhc', 'prec') |   ('fabriceyhc', 'rec') |   ('fabriceyhc', 'f1') |   ('wakaka', 'prec') |   ('wakaka', 'rec') |   ('wakaka', 'f1') |
|:---------|-------------------:|------------------:|-----------------:|-------------------------:|------------------------:|-----------------------:|-------------------------:|------------------------:|-----------------------:|---------------------:|--------------------:|-------------------:|
| AVG      |              0.892 |             0.892 |            0.892 |                    0.935 |                   0.935 |                  0.935 |                    0.933 |                   0.933 |                  0.933 |                0.905 |               0.905 |              0.905 |
| negative |              0.895 |             0.886 |            0.891 |                    0.931 |                   0.938 |                  0.935 |                    0.93  |                   0.935 |                  0.933 |                0.909 |               0.899 |              0.904 |
| positive |              0.888 |             0.897 |            0.893 |                    0.938 |                   0.931 |                  0.934 |                    0.935 |                   0.93  |                  0.933 |                0.901 |               0.91  |              0.906 |

## Credits:

* BERT Transformer by Huggingface: <https://huggingface.co/docs/transformers/model_doc/bert>
* NLP-Progress Scoreboard Sentiment Analysis (IMDb): <http://nlpprogress.com/english/sentiment_analysis.html#imdb>
