# Prediction-Constrained Topic Models
Public repo containing code to train, visualize, and evaluate semi-supervised topic models, as described in Hughes et al. AISTATS 2018

# Overview

# Examples

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


