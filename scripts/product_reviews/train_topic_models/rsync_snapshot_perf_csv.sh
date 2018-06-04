dataset_subpath="multi_domain_product_reviews_dataset/clean_data_v20180403/"
dataset_path="$HOME/git/$dataset_subpath/"
echo "dataset_path:"
echo "$dataset_path"

export XHOST_SSH_ADDR=mchughes@browncs
export XHOST_REMOTE_PATH="/nbu/liv/mhughes/slda_results/$dataset_subpath/"
export XHOST_LOCAL_PATH="/results/$dataset_subpath/"

bash $PC_REPO_DIR/scripts/rsync_tools/rsync_snapshot_perf_metrics.sh

