#!/bin/bash
#
# Common launcher script for ALL experiments
#
# Each call to this script will launch separate "job" (locally or on grid)
# Location of job depends on the value of XHOST environment variable

if [[ $XHOST == 'list' || $XHOST == 'dry' ]]; then
    if [[ -z $target_names ]]; then
        echo $output_path
    else
        echo $target_names $output_path
    fi
elif [[ $XHOST == 'grid' ]]; then
    # Launch each job on grid computing system (LSF/SLURM/SGE)
    launcher_exe=`python $PC_REPO_ROOT/scripts/grid_tools/detect_grid_executable.py`
    tmp_script_path=`python $PC_REPO_ROOT/scripts/grid_tools/make_launcher_script.py`
    CMD="$launcher_exe < $tmp_script_path"
    eval $CMD
elif [[ $XHOST == 'local' ]]; then
    # Launch each job on local cpu (same process that called launch_job.sh)
    echo $output_path
    bash $XHOST_BASH_EXE
    exit 1
elif [[ $XHOST == 'local_alltasks' ]]; then
    # Launch each job on local cpu (same process that called launch_job.sh)
    echo $output_path
    for XHOST_TASK_ID in `seq $XHOST_FIRSTTASK $XHOST_NTASKS`
    do
        echo ">>> task $XHOST_TASK_ID"
        export XHOST_TASK_ID=$XHOST_TASK_ID
        bash $XHOST_BASH_EXE
    done
    unset XHOST_TASK_ID
else
    if [[ -z $XHOST ]]; then 
        echo "ERROR: User did not define env variable: XHOST"
    else
        echo "ERROR: Unrecognized value for XHOST: $XHOST"
    fi
    echo "SUPPORTED OPTIONS:"
    echo "XHOST=list  : list output_path for all tasks, then exit"
    echo "XHOST=local : run first task on current local machine"
    echo "XHOST=local_alltasks : run all tasks serially on current local machine"
    echo "XHOST=grid  : run all tasks on available grid engine (SLURM/SGE/LSF)"
    exit 1
fi
