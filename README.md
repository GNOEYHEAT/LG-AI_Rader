# 자율주행 센서의 안테나 성능 예측 AI 경진대회

### Public score 2th 1.89526 | Private score 2th 1.91352

* 주최 : LG AI Research
* 주관 : DACON
* [https://dacon.io/competitions/official/235927/overview/description](https://dacon.io/competitions/official/235927/overview/description)

## Usage
- `train.sh` : If the trained model not exist, reproduce the model.
- `inference.sh` : If the trained model exist, only inference without training.

## Pipeline

### Setting Environment
- python version >= 3.6

#### Make virtual env
```
$ python3 -m venv pyenv
$ source ./pyenv/bin/activate
```

#### Install requirements
```
$ (pyenv) pip install --upgrade pip
$ (pyenv) pip install -r requirements.txt 
```

#### Train Run Shell
```
$ (pyenv) sh ./train.sh
```
or
```
$ (pyenv) python ./src/preprocess.py
$ (pyenv) python ./src/train.py
```

#### Inference Run Shell
```
$ (pyenv) sh ./inference.sh
```
or
```
$ (pyenv) python ./src/preprocess.py
$ (pyenv) python ./src/inference.py
```

### File Description
```
feature.py : feature engineering class py
model.py : model class py
preprocess.py : preprocess activate code
train.py : train activate code
utils.py : utils func py
inference.py : inference activate code
```

### Requirements
```
numpy
pandas
tqdm
scikit-learn
lightgbm==3.3.2
xgboost==1.6.1
catboost==1.0.6
```

## Directory Structure
<pre><code>
/workspace
├── model
│   ├── clean
│   │   ├── level0
│   │   ├── level1
│   ├── noise
│   │   ├── level0
│   │   ├── level1
├── open
│   ├── meta
│   │   ├── sample_submission.csv
│   │   ├── test.csv
│   │   ├── train.csv
├── output
│   ├── clean
│   │   ├── submission.csv
│   ├── final
│   │   ├── submission.csv
│   ├── noise
│   │   ├── submission.csv
├── refine
│   ├── clean
│   │   ├── raw
│   │   ├── scale
│   ├── noise
│   │   ├── raw
│   │   ├── scale
├── src
│   ├── feature.py
│   ├── inference.py
│   ├── model.py
│   ├── preprocess.py
│   ├── train.py
│   ├── utils.py
├── inference.sh
├── train.sh
      .
      .
      .
</code></pre>
