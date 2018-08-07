import argparse
import os
import time

import model_slda

from pc_toolbox.utils_io import (
    pprint,
    setup_output_path,
    setup_random_seed,
    write_user_provided_kwargs_to_txt,
    write_env_vars_to_txt,
    write_python_module_versions_to_txt,
    )

from pc_toolbox.algs_gradient_descent import (
    grad_descent_minimizer,
    scipy_lbfgs_minimizer,
    )

def train_slda_model(
        dataset_path=None,
        lossandgrad_mod_name=None,
        alg_name=None,
        frac_labels_train=1.0,
        n_states=None,
        alpha=None,
        tau=None,
        lambda_w=None,
        delta=None,
        weight_x=1.0,
        weight_y=0.0,
        n_batches=1,
        pi_max_iters_first_train_lap=1.0,
        **user_kwargs_P):

    # Load dataset
    datasets_by_split = dict()
    for split_name in ['train', 'valid', 'test']:
        datasets_by_split[split_name], data_info = \
            model_slda.slda_utils__dataset_manager.load_dataset(
                dataset_path=dataset_path,
                split_name=split_name,
                frac_labels_train=frac_labels_train,
                return_info=True)

    model_hyper_P = dict(
        K=int(n_states),
        alpha=float(alpha),
        tau=float(tau),
        lambda_w=float(lambda_w),        
        delta=delta, # only for non-binary outputs
        weight_x=weight_x,
        weight_y=weight_y,
        )
    dim_P = dict(
        n_vocabs=data_info['n_vocabs'],
        n_labels=data_info['n_labels'],
        n_states=int(n_states),
        )

    ## Load loss and gradient functions
    if lossandgrad_mod_name == 'slda_loss__autograd':
        mod = model_slda.slda_loss__autograd
    elif lossandgrad_mod_name == 'slda_loss__tensorflow':
        if model_slda.HAS_TENSORFLOW:
            mod = model_slda.slda_loss__tensorflow
        else:
            raise ValueError("slda_loss__tensorflow not available. Please install tensorflow.")
    else:
        raise ValueError("Unrecognized lossandgrad_mod_name: %s" % lossandgrad_mod_name)

    # Both loss and grad funcs take TWO input args
    # 1) param_vec (flattend, unconstrained values)
    # 2) step (which allows controling which subset/batch is used implicitly)
    loss_func, grad_func = mod.make_loss_func_and_grad_func_wrt_paramvec_and_step(
        dataset=datasets_by_split['train'],
        n_batches=int(n_batches),
        dim_P=dim_P,
        model_hyper_P=model_hyper_P,
        pi_max_iters_first_train_lap=int(pi_max_iters_first_train_lap),
        max_train_laps=float(user_kwargs_P['n_laps']),
        )

    # Initialize model params using common procedures
    init_param_dict = model_slda.slda_utils__init_manager.init_param_dict(
        dataset=datasets_by_split['train'],
        n_states=dim_P['n_states'],
        init_name=user_kwargs_P['init_name'],
        init_model_path=user_kwargs_P['init_model_path'],
        seed=user_kwargs_P['seed'],
        **model_hyper_P)

    # Setup perf_metrics_pi_optim_kwargs
    perf_metrics_pi_optim_kwargs = dict()
    for key in user_kwargs_P.keys():
        if key.startswith('perf_metrics_pi_'):
            std_key = key.replace('perf_metrics_', '')
            perf_metrics_pi_optim_kwargs[std_key] = user_kwargs_P.pop(key)

    # Load training algorithm
    if alg_name == 'grad_descent_minimizer':
        alg_mod = grad_descent_minimizer
    elif alg_name == 'scipy_lbfgs_minimizer':
        alg_mod = scipy_lbfgs_minimizer
    else:
        raise ValueError("Unrecognized alg_name: %s" % alg_name)
    param_dict, info_dict = alg_mod.minimize(
        loss_func_wrt_paramvec_and_step=loss_func,
        grad_func_wrt_paramvec_and_step=grad_func,
        save_func_wrt_param_dict=\
            model_slda.slda_utils__param_io_manager.save_topic_model_param_dict,
        init_param_dict=init_param_dict,
        param_tfm_manager=model_slda.slda_utils__param_manager,
        dim_P=dim_P,
        callback_func_wrt_param_dict=\
            model_slda.slda_snapshot_perf_metrics.\
                calc_perf_metrics_for_snapshot_param_dict,
        callback_kwargs=dict(
            datasets_by_split=datasets_by_split,
            model_hyper_P=model_hyper_P,
            dim_P=dim_P,
            perf_metrics_pi_optim_kwargs=perf_metrics_pi_optim_kwargs,
            ),
        n_batches=n_batches,
        **user_kwargs_P)
    return param_dict, info_dict

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=os.path.expandvars('$PC_REPO_DIR/datasets/toy_bars_3x3/'),
        )
    parser.add_argument(
        '--lossandgrad_mod_name',
        type=str,
        default='slda_loss__autograd',
        choices=['slda_loss__autograd', 'slda_loss__tensorflow'],
        help="Name of module that defines loss function and its gradients")
    parser.add_argument(
        '--alg_name',
        type=str,
        default='grad_descent_minimizer',
        choices=['grad_descent_minimizer', 'scipy_lbfgs_minimizer'],
        help="Name of module that performs optimization on loss function")
    parser.add_argument(
        '--n_states',
        type=int,
        default=1,
        help="Number of states/topics for the model.")
    parser.add_argument(
        '--n_laps',
        type=float,
        default=2,
        help="Number of passes through training dataset to complete.")
    parser.add_argument(
        '--n_batches',
        type=int,
        default=1,
        help="Number of batches to divide training dataset into.")
    parser.add_argument(
        '--frac_labels_train',
        type=float,
        default=1.0,
        help=(
            "Fraction of training examples to keep when training."
            + " Set to <1.0 to simulate semi-supervised settings."))
    ###
    # grad_descent_minimizer specific alg settings
    parser.add_argument(
        '--step_direction',
        type=str,
        default='adam',
        choices=['adam', 'steepest'])
    parser.add_argument(
        '--step_size',
        type=float,
        default=0.5,
        help="Step size / learning rate for gradient descent.")
    parser.add_argument(
        '--decay_rate',
        type=float,
        default=1.0,
        help='Rate at which step_size decays over steps in gradient descent.')
    parser.add_argument(
        '--decay_interval',
        type=float,
        default=25.0)
    parser.add_argument(
        '--decay_staircase',
        type=int,
        default=0,
        help="0 = decay of step_size happens smoothly. 1 = staircase.")
    parser.add_argument(
        '--max_l2_norm_of_grad_per_entry',
        type=float,
        default=10.0,
        help="Enforce gradient clipping if l2 norm of gradient exceeds this value.")

    ###
    # sLDA model hyperparameters
    parser.add_argument(
        '--alpha', 
        type=float,
        default=1.01,
        help="Document-topic concentration parameter. Should be >1.0.")
    parser.add_argument(
        '--lambda_w',
        type=float,
        default=0.001)
    parser.add_argument(
        '--tau',
        type=float,
        default=1.01)
    parser.add_argument(
        '--delta',
        type=float,
        default=1.0,
        help='variance of regression predictions')

    ###
    # sLDA loss function parameters
    parser.add_argument(
        '--weight_x',
        type=float,
        default=1.0)
    parser.add_argument(
        '--weight_y',
        type=float,
        default=1.0)

    ###
    # sLDA initialization parameters
    parser.add_argument(
        '--init_name',
        type=str,
        default='rand_docs')
    parser.add_argument(
        '--init_model_path',
        type=str,
        default=None)

    ###
    # Per-doc estimation kwargs
    pi_optim_kwargs = model_slda.\
        est_local_params__single_doc_map.DefaultDocTopicOptKwargs
    parser.add_argument(
        '--pi_max_iters_first_train_lap',
        type=float,
        default=pi_optim_kwargs['pi_max_iters'],
        help=(
            "Max iters for pi estimation allowed on first lap."
            + " Will gradually ramp up to pi_max_iters after 50% of laps"))
    for key in sorted(pi_optim_kwargs.keys()):
        default_val = pi_optim_kwargs[key]
        parser.add_argument(
            '--perf_metrics_' + key,
            type=type(default_val),
            default=default_val,
            )

    ###
    # Random seed
    parser.add_argument(
        '--seed',
        type=str,
        default="from_output_path_and_taskid",
        help=(
            "Random seed for initialization of model parameters."
            + " Default value determines seed directly from output_path."))

    ### 
    # Output format settings
    parser.add_argument(
        '--output_path',
        type=str,
        default=None)
    parser.add_argument(
        '--param_output_fmt',
        type=str,
        default='dump')
    parser.add_argument(
        '--n_seconds_between_save',
        type=float,
        default=-1)
    parser.add_argument(
        '--n_seconds_between_print',
        type=float,
        default=-1)
    parser.add_argument(
        '--n_steps_between_save',
        type=float,
        default=1)
    parser.add_argument(
        '--n_steps_between_print',
        type=float,
        default=1)
    parser.add_argument(
        '--n_steps_to_print_early',
        type=float,
        default=5)
    parser.add_argument(
        '--n_steps_to_save_early',
        type=float,
        default=5)

    ### Parse that input!
    args, unk_args = parser.parse_known_args()
    unk_dict = dict(
        [(k[2:], v) for (k, v) in zip(unk_args[::2], unk_args[1::2])]
        )
    arg_dict = vars(args)
    arg_dict.update(unk_dict)
    arg_dict['output_path'] = setup_output_path(
        **arg_dict)
    arg_dict['seed'] = setup_random_seed(
        **arg_dict)

    # Write useful environment info to .txt
    # so we can reproduce later
    write_user_provided_kwargs_to_txt(
        arg_dict=arg_dict,
        output_path=arg_dict['output_path'])
    write_env_vars_to_txt(
        output_path=arg_dict['output_path'])
    write_python_module_versions_to_txt(
        context_dict=locals(),
        output_path=arg_dict['output_path'])

    train_slda_model(
        **arg_dict)