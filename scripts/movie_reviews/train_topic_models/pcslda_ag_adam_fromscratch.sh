export XHOST_NTASKS=3
export XHOST_BASH_EXE=$PC_REPO_DIR/scripts/train_slda.sh
nickname=20180301

export lossandgrad_mod_name="slda_loss__autograd"

# =============================== DATA SETTINGS
export dataset_name=movie_reviews_pang_lee
export dataset_path="$PC_REPO_DIR/datasets/$dataset_name/"
export n_vocabs=5338
export n_train_docs=4004
export frac_labels_train=1.000

for n_batches in 1 5
do
    export n_batches=$n_batches

# =============================== OUTPUT SETTINGS
export param_output_fmt="topic_model_snapshot"
n_laps_btw=25
n_steps_btw=`python -c "print $n_batches * $n_laps_btw"`
export n_steps_between_save=$n_steps_btw
export n_steps_between_print=$n_steps_btw
export n_steps_to_print_early=2
export n_steps_to_save_early=2
export laps_to_save_custom='0,1,2,5,10,15,20'

# =============================== ALGO SETTINGS
export n_laps=500

## Overall training: ADAM 
export alg_name="grad_descent_minimizer"
export step_direction='adam'
export decay_staircase=0
export decay_interval=1
export decay_rate=0.997
for step_size in 0.0333 0.3333
do
    export step_size=$step_size

## Per-doc inference settings
export pi_max_iters=100
export pi_step_size=0.05
# Try to make early iterations faster by doing less per-doc work
# Will gradually ramp up from ___ to pi_max_iters
for pi_max_iters_first_train_lap in 010 100
do
    export pi_max_iters_first_train_lap=$pi_max_iters_first_train_lap

# =============================== INIT SETTINGS
export init_model_path=none
for init_name in rand_smooth
do
    export init_name=$init_name

# =============================== MODEL HYPERS
export alpha=1.100
export tau=1.100
export lambda_w=0.001

export weight_x=1.0

## Loop over weights to place on log p(y|x)
for weight_y in 100.0 010.0 001.0 #000.0
do
    export weight_y=$weight_y

## Loop over number of topics K
for n_states in 010 025 050
do
    export n_states=$n_states

    export output_path="$XHOST_RESULTS_DIR/$dataset_name/$nickname-n_batches=$n_batches-lossandgrad_mod=$lossandgrad_mod_name-n_states=$n_states-alpha=$alpha-tau=$tau-lambda_w=$lambda_w-weight_x=$weight_x-weight_y=$weight_y-init_name=$init_name-alg_name=$step_direction-step_size=$step_size-pi_iters_first_lap=$pi_max_iters_first_train_lap/1/"

    bash $PC_REPO_DIR/scripts/launch_job_on_host_via_env.sh || { exit 1; }

done
done
done
done
done
