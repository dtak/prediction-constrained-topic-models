# Steps to download training results to local computer and make plots

## 1) On remote machines, do intensive training

Typically, you'll run some pc_toolbox training algos on big datasets for several hours or days.

These will dump results onto disk on the remote file system, in a path like

    XHOST_REMOTE_PATH/20180301-<some info>/1/snapshot_perf_metrics_train.csv

We'll use `XHOST_REMOTE_PATH` to denote the value of XHOST_RESULTS_DIR on the remote system.

## 2) On local machine, run "rsync_snapshot_perf_metrics.sh" to grab the results

```
$ cd $PC_REPO_DIR/scripts/rsync_tools/

# SET UP YOUR SSH INFO
$ export SSH_ADDR=<user>@<domain.edu>

# SET REMOTE PATH TO DIRECTORY THAT CONTAINS FILES FOR DATASET OF INTEREST
$ export XHOST_REMOTE_PATH=/remote_path/to/results/

# SET LOCAL PATH ON YOUR MACHINE
$ export XHOST_LOCAL_PATH=/local_path/to/results/

# RUN RSYNC SCRIPT
$ bash rsync_snaphsot_perf_metrics.sh
```

#### Expected output:

```
$ bash rsync_snapshot_perf_metrics.sh 
receiving file list ... 
24 files to consider
<lots more>
```

## 3) On local machine, run jupyter notebook to plot results

#### **in bash**
$ cd /path/to/notebooks/
$ jupyter notebook

#### **in jupyter notebook**


