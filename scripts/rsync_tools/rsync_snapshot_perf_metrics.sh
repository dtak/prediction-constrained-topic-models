if [[ -z $XHOST_SSH_ADDR ]]; then
    echo "ERROR: Define $XHOST_SSH_ADDR=<user>@<domain.edu>"
    exit;
fi

if [[ -z $XHOST_REMOTE_PATH ]]; then
    XHOST_REMOTE_PATH=/nbu/liv/mhughes/public_results/toy_bars_3x3/
fi

if [[ -z $XHOST_LOCAL_PATH ]]; then
    XHOST_LOCAL_PATH=/tmp/toy_bars_3x3/
fi
mkdir -p $XHOST_LOCAL_PATH

# Ask rsync to copy all files from remote to local
# which match the specific .csv file template

rsync -armPKL \
    --include="/*/*/snapshot_perf_metrics_*.csv" \
    --exclude="/*/*/*" \
    $XHOST_SSH_ADDR:$XHOST_REMOTE_PATH \
    $XHOST_LOCAL_PATH/

