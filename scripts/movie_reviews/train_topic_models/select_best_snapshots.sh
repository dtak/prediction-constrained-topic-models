dataset_subpath="movie_reviews_pang_lee/"
dataset_path="$PC_REPO_DIR/datasets/$dataset_subpath"
export XHOST_SSH_ADDR=mchughes@browncs
export XHOST_REMOTE_PATH="/nbu/liv/mhughes/slda_results/$dataset_subpath/"
export XHOST_LOCAL_PATH="/results/$dataset_subpath/"

results_path_pattern_01="$XHOST_LOCAL_PATH/20180301*tensorflow*"

output_path="$XHOST_LOCAL_PATH/best_runs_20180301_pcslda_tensorflow/"

python $PC_REPO_DIR/pc_toolbox/utils_snapshots/select_best_runs_and_snapshots.py \
    --output_path $output_path \
    --legend_name PC_sLDA \
    --results_path_patterns "$results_path_pattern_01" \
    --txt_src_path $dataset_path \
    --target_y_name more_than_2_out_of_4_stars \
    --all_y_names more_than_2_out_of_4_stars \
    --selection_score_colname Y_ROC_AUC \
    --selection_score_ranking_func argmax \
    --col_names_to_use_at_selection N_STATES,WEIGHT_Y \
    --col_names_to_keep_per_split \
        Y_ROC_AUC,Y_ERROR_RATE,LOGPDF_X_PERTOK,LOGPDF_Y_PERDOC \
    --col_names_to_keep \
        ALPHA,TAU,LAMBDA_W \