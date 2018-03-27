export XHOST_SSH_ADDR=mchughes@odyssey
export XHOST_REMOTE_PATH=/n/dtak/mchughes/slda_results/bow_pang_movie_reviews/v20170929_split_80_10_10/
export XHOST_LOCAL_PATH=/results/bow_pang_movie_reviews_v20170929/


#pushd $SSCAPEROOT/notebooks/html_interp_viz/
pushd $SSCAPEROOT/externals/topic_models/utils_viz_topic_models/

for show_enriched_words in 0 1
do

python make_html_collection_from_csv.py \
    --snapshot_csv_path $XHOST_LOCAL_PATH/per_snapshot_srcfiles.csv \
    --html_output_path /tmp/html_camready/movies/rerank_words="$show_enriched_words"/ \
    --field_order LEGEND_NAME,LABEL_NAME,FRAC_LABELS,N_STATES \
    --ncols 4 \
    --n_chars_per_word 20 \
    --n_words_per_topic 15 \
    --show_enriched_words $show_enriched_words \
    --show_longer_words_via_tooltip 1 \

done

# return home
popd