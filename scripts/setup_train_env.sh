#!/bin/bash

echo "---------- START $0"

if [[ -z $PC_REPO_DIR ]]; then
    echo "Error: Need to define PC_REPO_DIR." 1>&2;
    exit;
fi
echo "PC_REPO_DIR=$PC_REPO_DIR"

if [[ -z $output_path ]]; then
    echo "Error: Need to define output_path." 1>&2;
fi
echo "output_path:"
echo $output_path

export PYTHONPATH="$PC_REPO_DIR:$PYTHONPATH"
echo "PYTHONPATH=$PYTHONPATH"

# Set default: single threaded
if [[ -z $OMP_NUM_THREADS ]]; then
    export OMP_NUM_THREADS=1
fi
if [[ -z $MKL_NUM_THREADS ]]; then
    export MKL_NUM_THREADS=1
fi

# Set default: which python executable to use
if [[ -z $XHOST_PYTHON_EXE ]]; then
    export XHOST_PYTHON_EXE=`which python`
fi
echo "Python executable:"
echo $XHOST_PYTHON_EXE

# If user desired to run on grid computing...
if [[ $XHOST == 'grid' ]]; then
    # Verify place to write logs exists
    if [[ -z $XHOST_LOG_DIR ]]; then
        echo "Error: Need to define XHOST_LOG_DIR." 1>&2;
        exit;
    fi
    echo "XHOST_LOG_DIR=$XHOST_LOG_DIR"


    # Avoid race conditions on NFS file access
    # by sleeping a little while (60 sec or less)
    sleep $[ ( $RANDOM % 60 )  + 1 ]s
fi

echo "----------  STOP $0"