# Prediction-Constrained Topic Models
Public repo containing code to train, visualize, and evaluate semi-supervised topic models, as described in Hughes et al. AISTATS 2018:

# Overview

### Repo overview
```
* datasets/
  Provided datasets for simple experiments
* pc_toolbox/
  Main python package
* scripts/
  Bash scripts to run experiments
```

### Resources

* Paper PDF: https://www.michaelchughes.com/papers/HughesEtAl_AISTATS_2018.pdf
* Supplement PDF: https://www.michaelchughes.com/papers/HughesEtAl_AISTATS_2018_supplement.pdf

# Examples

## Basic script

See train_slda_model.py
```
python train_slda_model.py \
  --dataset_path $PC_REPO_DIR/datasets/toy_bars_3x3/ \
  --output_path /tmp/ \
  --seed 8675309 \
  --alpha 1.1 \
  --tau 1.1 \
  --weight_y 5.0 \        # aka "lambda" in AISTATS paper, the key hyperparameter to emphasize y|x
  --n_laps 10 \
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


