# Movie Reviews dataset (Pang and Lee 2005)

Raw text from movie reviews of four critics comes from scaledata v1.0 dataset released by Pang and Lee (http://www.cs.cornell.edu/people/pabo/movie-review-data/).

## Preprocessing

Given plain text files of movie reviews, we tokenized and then stemmed using the Snowball stemmer from the nltk Python package, so that words with similar roots (e.g. film, films, filming) all become the same token. We removed all tokens in Mallet's list of common English stop words as well as any token included in the 1000 most common first names from the US census. We added this step after seeing too many common first names like Michael and Jennifer appear meaninglessly in many top-word lists for trained topics. We manually whitelisted "oscar" and "tony" due to their saliency to movie reviews sentiment. We then performed counts of all remaining tokens across the full raw corpus of 5006 documents, discarding any tokens that appear at least once in more than 20\% of all documents or less than 30 distinct documents. The final vocabulary list has 5375 terms.

Each of the 5006 original documents was then reduced to this vocabulary set. We discarded any documents that were too short (less than 20 tokens), leaving 5005 documents. Each document has a binary label, where 0 indicates it has a negative review (below 0.6 in the original datasets' 0-1 scale) and 1 indicates positive review (>= 0.6). This 0.6 threshold matches a threshold previously used in the raw data's 4-category scale to separate 0 and 1 star reviews from 2 and 3 (of 3) star reviews. Data pairs ($x_d, y_d$) were then split into training, validation, test. Both validation and test used 10 \% of all documents, evenly balancing positive and negative labeled documents. The remaining documents were allocated to the training set.


## Dataset Specs

Specs computed via
```
python $PC_REPO_DIR/pc_toolbox/model_slda/slda_utils__dataset_manager.py \
    --dataset_path $PC_REPO_DIR/datasets/movie_reviews_pang_lee/ \
    --dataset_name movie_reviews
```

### TRAIN set of movie_reviews

```
4004 docs
5338 vocab words
unique tokens per doc   0%:     29    1%:     69   10%:    103   50%:    151   90%:    205   99%:    295  100%:    438
 total tokens per doc   0%:     29    1%:     77   10%:    120   50%:    183   90%:    260   99%:    403  100%:    644
1 labels
1.000 (4004/4004) docs are labeled
 more_than_2_out_of_4_stars ( 1/1) frac positive 0.578 (  2315/4004)
```

### VALID set of movie_reviews
```
500 docs
5338 vocab words
unique tokens per doc   0%:     33    1%:     64   10%:    107   50%:    153   90%:    213   99%:    296  100%:    462
 total tokens per doc   0%:     46    1%:     76   10%:    125   50%:    189   90%:    272   99%:    416  100%:    780
1 labels
1.000 (500/500) docs are labeled
 more_than_2_out_of_4_stars ( 1/1) frac positive 0.498 (   249/500)

### TEST set of movie_reviews
```
501 docs
5338 vocab words
unique tokens per doc   0%:     37    1%:     74   10%:    101   50%:    146   90%:    206   99%:    300  100%:    405
 total tokens per doc   0%:     39    1%:     84   10%:    119   50%:    177   90%:    264   99%:    406  100%:    621
1 labels
1.000 (501/501) docs are labeled
 more_than_2_out_of_4_stars ( 1/1) frac positive 0.547 (   274/501)
```
