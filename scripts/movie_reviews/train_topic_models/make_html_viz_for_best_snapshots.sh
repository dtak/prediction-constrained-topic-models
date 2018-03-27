
pushd $PC_REPO_DIR/pc_toolbox/utils_vizhtml/

template_html='<pre>TRAIN AUC=$TRAIN_Y_ROC_AUC<br />VALID AUC=$VALID_Y_ROC_AUC<br /> TEST AUC=$TEST_Y_ROC_AUC<nbsp;></pre>'

for rank_words_by in 'proba_word_given_topic' 'proba_topic_given_word'
do
python make_html_collection_from_csv.py \
    --snapshot_csv_path $XHOST_LOCAL_PATH/best_runs_20180301_pcslda_tensorflow/best_snapshots_PC_sLDA.csv \
    --html_output_path /tmp/movie_reviews_html/rank_words_by="$rank_words_by"/ \
    --field_order LEGEND_NAME,LABEL_NAME,N_STATES,WEIGHT_Y \
    --ncols 4 \
    --n_chars_per_word 20 \
    --n_words_per_topic 15 \
    --rank_words_by $rank_words_by \
    --show_longer_words_via_tooltip 1 \
    --metrics_template_html "$template_html" \

done


popd