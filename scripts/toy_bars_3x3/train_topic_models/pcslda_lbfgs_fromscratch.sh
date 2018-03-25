export XHOST_NTASKS=3
export XHOST_BASH_EXE=$PC_REPO_DIR/scripts/train_slda.sh
nickname=20180301

export lossandgrad_mod_name="slda_loss__autograd"

# =============================== DATA SETTINGS
export dataset_name=toy_bars_3x3
export dataset_path="$PC_REPO_DIR/datasets/$dataset_name/"
export n_vocabs=9
export n_outputs=2
export n_train_docs=500

export n_batches=1

# =============================== OUTPUT SETTINGS
export param_output_fmt="topic_model_snapshot"
export n_steps_between_save=10
export n_steps_between_print=10
export n_steps_to_print_early=2
export n_steps_to_save_early=2
export laps_to_save_custom='0,1,2,4,6,8,10'

# =============================== ALGO SETTINGS
export n_laps=100

## Overall training: L-BFGS 
export alg_name="scipy_lbfgs_minimizer"

## Per-doc inference settings
export pi_max_iters=100
export pi_step_size=0.05

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
for weight_y in 100.0 010.0 001.0
do
    export weight_y=$weight_y

for n_states in 004
do
    export n_states=$n_states

    export output_path="$XHOST_RESULTS_DIR/$dataset_name/$nickname-n_batches=$n_batches-lossandgrad_mod=$lossandgrad_mod_name-n_states=$n_states-alpha=$alpha-tau=$tau-lambda_w=$lambda_w-init_name=$init_name-alg_name=$alg_name-weight_x=$weight_x-weight_y=$weight_y/1/"

    bash $SSCAPEROOT/scripts/launch_job_on_host_via_env.sh || { exit 1; }

done
done
done
