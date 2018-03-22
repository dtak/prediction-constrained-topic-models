#!/bin/bash

echo "---------- START $0"

# Setup env vars for this training run
. $PC_REPO_DIR/scripts/setup_train_env.sh
XHOST_SCRIPT=$PC_REPO_DIR/pc_toolbox/scripts/train_model.py

# Parse keyword args from env
keyword_args=`python $PC_REPO_DIR/scripts/launcher_tools/print_lowercase_env_vars_as_keyword_args.py`
echo "SCRIPT BASENAME:"
echo `basename $XHOST_SCRIPT`
echo "SCRIPT PATH:"
echo $XHOST_SCRIPT
echo "SCRIPT KWARGS:"
echo $keyword_args

# Run desired script
eval $XHOST_PYTHON_EXE -u $XHOST_SCRIPT $keyword_args

echo "----------  STOP $0"
