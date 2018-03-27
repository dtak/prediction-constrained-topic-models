#!/bin/bash 
#
# rsync_specific_snapshot.sh : Remote sync model snapshots.
#
# Usage
# -----
# $ bash rsync_specific_snapshot.sh <remote_snapshot_path>
#
# Will rsync all content of that remote path to provided XHOST_LOCAL_PATH.

# Parse REMOTE_SNAPSHOT_PATH
if [[ -z $1 ]]; then
    echo "ERROR: Missing remote snapshot path arg;"
    exit;
fi
REMOTE_SNAPSHOT_PATH=$1

if [[ -z $XHOST_SSH_ADDR ]]; then
    XHOST_SSH_ADDR=mhughes@ssh.cs.brown.edu
fi

if [[ -z $XHOST_REMOTE_PATH ]]; then
    XHOST_REMOTE_PATH=/nbu/liv/mhughes/public_results/bow_toy_letters/
fi
XHOST_REMOTE_PATH=`python -c "import os; print '$XHOST_REMOTE_PATH'.rstrip(os.path.sep) + '/'"`

if [[ -z $XHOST_LOCAL_PATH ]]; then
    XHOST_LOCAL_PATH=/tmp/bow_toy_letters/
fi
XHOST_LOCAL_PATH=`python -c "import os; print '$XHOST_LOCAL_PATH'.rstrip(os.path.sep) + '/'"`

# Force REMOTE_SNAPSHOT_PATH to look like directory, if desired
REMOTE_SNAPSHOT_PATH=`python -c "print '$REMOTE_SNAPSHOT_PATH'.replace('$XHOST_LOCAL_PATH', '$XHOST_REMOTE_PATH')"`
IS_DIR=`python -c "print '$REMOTE_SNAPSHOT_PATH'.endswith('/')"`
if [[ $IS_DIR == 'True' ]]; then
    REMOTE_SNAPSHOT_PATH=`python -c "import os; print '$REMOTE_SNAPSHOT_PATH'.rstrip(os.path.sep) + '/'"`
fi

echo "START rsync_specific_snapshot.sh"
echo ">>> IS_DIR=$IS_DIR"
echo ">>> XHOST_SSH_ADDR=$XHOST_SSH_ADDR"
echo ">>> XHOST_REMOTE_PATH=$XHOST_REMOTE_PATH"
echo ">>> XHOST_LOCAL_PATH=$XHOST_LOCAL_PATH"

if [[ -z $2 ]]; then
    LOCAL_SNAPSHOT_PATH=`python -c "print '$REMOTE_SNAPSHOT_PATH'.replace('$XHOST_REMOTE_PATH', '$XHOST_LOCAL_PATH')"`
else
    LOCAL_SNAPSHOT_PATH=$2;
fi

# Copy any files in the provided snapshot folder
# to the local snapshot folder
if [[ $IS_DIR == 'True' ]]; then
    # This branch needs the trailing /
    echo ">>> REMOTE_SNAPSHOT_PATH=$REMOTE_SNAPSHOT_PATH"
    echo ">>> LOCAL_SNAPSHOT_PATH=$LOCAL_SNAPSHOT_PATH"

    # Avoid "yes/no" question to "are you sure you trust...?"
    #    -e "ssh -o StrictHostKeyChecking=no" \
    rsync -armPKL \
        $XHOST_SSH_ADDR:$REMOTE_SNAPSHOT_PATH/ \
        $LOCAL_SNAPSHOT_PATH/

else
    BASENAME=`python -c "import os; print os.path.split('$REMOTE_SNAPSHOT_PATH')[-1]"`
    REMOTE_SNAPSHOT_PATH=`python -c "import os; print os.path.split('$REMOTE_SNAPSHOT_PATH')[0]"`

    LOCAL_SNAPSHOT_PATH=`python -c "print '$REMOTE_SNAPSHOT_PATH'.replace('$XHOST_REMOTE_PATH', '$XHOST_LOCAL_PATH')"`

    echo $REMOTE_SNAPSHOT_PATH/
    echo $LOCAL_SNAPSHOT_PATH/
    echo $BASENAME

    scp $XHOST_SSH_ADDR:$REMOTE_SNAPSHOT_PATH/$BASENAME $LOCAL_SNAPSHOT_PATH/
    #rsync -armPKL \
    #    $XHOST_SSH_ADDR:$REMOTE_SNAPSHOT_PATH/ \
    #    $LOCAL_SNAPSHOT_PATH/ \
    #    --include='$BASENAME' \
    #    --exclude='*' \

fi


