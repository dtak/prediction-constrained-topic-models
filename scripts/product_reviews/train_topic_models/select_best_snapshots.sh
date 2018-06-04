dataset_subpath="multi_domain_product_reviews_dataset/clean_data_v20180403/"
dataset_path="$HOME/git/$dataset_subpath/"
echo "dataset_path:"
echo "$dataset_path"

export XHOST_SSH_ADDR=mchughes@browncs
export XHOST_REMOTE_PATH="/nbu/liv/mhughes/slda_results/$dataset_subpath/"
export XHOST_LOCAL_PATH="/results/$dataset_subpath/"

y_colnames=`cat $dataset_path/Y_colnames.txt`
echo "target_y_name: $y_colnames"

results_path_pattern_01="$XHOST_LOCAL_PATH/20180301*tensorflow*"
output_path="$XHOST_LOCAL_PATH/best_runs_20180301_pcslda_tensorflow/"

python $PC_REPO_DIR/pc_toolbox/utils_snapshots/select_best_runs_and_snapshots.py \
    --output_path $output_path \
    --legend_name PC_sLDA \
    --results_path_patterns "$results_path_pattern_01" \
    --txt_src_path $dataset_path \
    --target_y_name $y_colnames \
    --all_y_names $y_colnames \
    --selection_score_colname Y_ROC_AUC \
    --selection_score_ranking_func argmax \
    --col_names_to_use_at_selection N_STATES,WEIGHT_Y \
    --col_names_to_keep_per_split \
        Y_ROC_AUC,Y_ERROR_RATE,LOGPDF_X_PERTOK,LOGPDF_Y_PERDOC \
    --col_names_to_keep \
        ALPHA,TAU,LAMBDA_W,N_BATCHES \
