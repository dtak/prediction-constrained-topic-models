# Prediction-Constrained Topic Models

Public repo containing code to train, visualize, and evaluate semi-supervised topic models. Also includes code for baseline classifiers/regressors to perform supervised prediction on bag-of-words datasets.

# Overview

This repo is based on the following academic publication:

> "Prediction-constrained semi-supervised topic models"
> M. C. Hughes, L. Weiner, G. Hope, T. H. McCoy, R. H. Perlis, E. B. Sudderth, and F. Doshi-Velez
> Artificial Intelligence & Statistics (AISTATS), 2018.

* Paper PDF: https://www.michaelchughes.com/papers/HughesEtAl_AISTATS_2018.pdf
* Supplement PDF: https://www.michaelchughes.com/papers/HughesEtAl_AISTATS_2018_supplement.pdf

### Contents

* [datasets/](https://github.com/dtak/prediction-constrained-topic-models/tree/master/datasets/)
* * Provided example datasets for simple experiments. Overview in [datasets/README.md](https://github.com/dtak/prediction-constrained-topic-models/tree/master/datasets/README.md).
  
* [pc_toolbox/](https://github.com/dtak/prediction-constrained-topic-models/tree/master/pc_toolbox/)
* * Main python package, with code for training PC topic models and some baseline classifiers/regressors.

* [scripts/](https://github.com/dtak/prediction-constrained-topic-models/tree/master/scripts/)
* * Bash scripts to run experiments. Support SLURM/LSF/SGE clusters.


# Examples

## Python script to train binary classifier from bag-of-words data

The primary script is train_and_eval_sklearn_binary_classifier.py
```
python train_and_eval_sklearn_binary_classifier.py \
  --dataset_path $PC_REPO_DIR/datasets/toy_bars_3x3/ \
  --output_path /tmp/demo_results/ \
  --seed 8675309 \        # random seed (for reproducibility)
  --feature_arr_names X \
  --target_arr_name Y \
  --classifier_name extra_trees \
```


## Python script to train topic models with PC objective

The primary script is train_slda_model.py. For a quick exmaple, you might call this python script as follows:

```
python train_slda_model.py \
  --dataset_path $PC_REPO_DIR/datasets/toy_bars_3x3/ \
  --output_path /tmp/demo_results/ \
  --seed 8675309 \        # random seed (for reproducibility)
  --alpha 1.1 \           # scalar hyperparameter for Dirichlet prior over doc-topic probas
  --tau 1.1 \             # scalar hyperparameter for Dirichlet prior over topic-word probas
  --weight_y 5.0 \        # aka "lambda" in AISTATS paper, the key hyperparameter to emphasize y|x
  --n_laps 10 \           # number of laps (aka epochs). this will complete 10 full passes thru training dataset.
  --n_batches 1 \
  --alg_name grad_descent_minimizer \
```

Mostly, we use wrapper bash scripts that call this function with many different hyperparameters (model, algorithm, initialization, etc)

## Quicktest: Train PC sLDA topic models on toy bars with autograd

Test script to train with autograd (ag) as source of automatic gradients:
```
cd $PC_REPO_DIR/scripts/toy_bars_3x3/quicktest_topic_models/
export XHOST_RESULTS_DIR=/tmp/
XHOST=local bash pcslda_ag_adam_fromscratch.sh 
```
Should finish in <1 minute, just demonstrate that training occurs without errors.

## Quicktest: Train PC sLDA topic models on toy bars with tensorflow

Test script to train with tensorflow (tf) as source of automatic gradients:
```
cd $PC_REPO_DIR/scripts/toy_bars_3x3/quicktest_topic_models/
export XHOST_RESULTS_DIR=/tmp/
XHOST=local bash pcslda_tf_adam_fromscratch.sh 
```
Should finish in <1 minute, just demonstrate that training occurs without errors.


## Train PC sLDA topic models extensively on movie_reviews dataset

Script to train with tensorflow (tf) as source of automatic gradients:
```
cd $PC_REPO_DIR/scripts/movie_reviews_pang_lee/train_topic_models/
export XHOST_RESULTS_DIR=/tmp/
XHOST={local|grid} bash pcslda_tf_adam_fromscratch.sh 
```
Use XHOST=local to run on local computer.
Use XHOST=grid to launch jobs on a cluster (Sun Grid Engine, SLURM, IBM's LSF, etc).

Should finish in a few hours.


# Installation

* Step 1: Clone this repo

git clone https://github.com/dtak/prediction-constrained-topic-models/

* Step 2: Setup a fresh conda enviroment with all required Python packages

bash [`$PC_REPO_DIR/scripts/install/create_conda_env.sh`](https://github.com/dtak/prediction-constrained-topic-models/tree/master/scripts/install/create_conda_env.sh)

* Step 3: Compile Cython code for per-document inference (makes things very fast)

`cd $PC_REPO_DIR/`

python [`setup.py`](https://github.com/dtak/prediction-constrained-topic-models/tree/master/setup.py) `build_ext --inplace`

* Step 4: (Optional) Install tensorflow

bash [`$PC_REPO_DIR/scripts/install/install_tensorflow_linux.sh`](https://github.com/dtak/prediction-constrained-topic-models/tree/master/scripts/install/install_tensorflow_linux.sh)

# Configuration

Set up your environment variables!

First, make a shortcut variable to the location of this repo, so you can easily reference datasets, etc.

    $ export PC_REPO_DIR=/path/to/prediction_constrained_topic_models/

Second, add this repo to your python path, so you can do "import pc_toolbox"

    $ export PYTHONPATH=$PC_REPO_DIR:$PYTHONPATH

