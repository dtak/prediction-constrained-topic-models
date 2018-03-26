export XHOST_NTASKS=1
export XHOST_BASH_EXE=$PC_REPO_DIR/scripts/train_clf.sh
nickname="20180301"

# =============================== DATA SETTINGS
export dataset_name=movie_reviews_pang_lee
export dataset_path="$PC_REPO_DIR/datasets/$dataset_name/"

export feature_arr_names=X
export target_arr_name=Y

all_target_names=`cat "$dataset_path/Y_colnames.txt"`

for classifier_name in logistic_regression extra_trees
do

for target_name in $all_target_names
do
    export target_names=$target_name
    export classifier_name=$classifier_name
    export class_weight_opts='none'
    export preproc_X='none'

    export output_path="$XHOST_RESULTS_DIR/$dataset_name/$nickname-classifier_name=$classifier_name/1/"
    bash $PC_REPO_DIR/scripts/launch_job_on_host_via_env.sh || { exit 1; }

done
done
