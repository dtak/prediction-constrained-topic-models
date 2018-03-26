#!/bin/bash

export PC_ENV_NAME=pc_toolbox_env

conda create --name $PC_ENV_NAME python=2.7

source activate $PC_ENV_NAME

# Read requirements and install one line at a time
# For each req, we first try to do a conda install
# Falling back on 'pip' if conda doesnt work
while read requirement; do
    echo ">>> install $requirement START"
    conda install --yes $requirement || pip install $requirement;
    echo ">>> install $requirement DONE"
done < requirements.txt


