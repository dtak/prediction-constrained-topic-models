#!/bin/bash

#conda create --name pc_toolbox_env python=2.7

source activate pc_toolbox_env || echo "Error: check PATH";
echo "$PATH"

while read requirement; do
    conda install --yes $requirement || pip install $requirement;
done < requirements.txt


