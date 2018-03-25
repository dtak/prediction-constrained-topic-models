import argparse
import numpy as np
import time

import scipy.optimize

from pc_toolbox.utils_io import (
    pprint,
    do_print_now,
    do_save_now,
    default_settings_alg_io,
    init_alg_state_kwargs,
    update_alg_state_kwargs,
    make_status_string,
    save_status_to_txt_files,
    append_to_txtfile,
    update_alg_state_kwargs_after_print,
    update_alg_state_kwargs_after_save,
    calc_laps_when_snapshots_saved,
    )

from grad_descent_minimizer import calc_l2_norm_of_vector_per_entry

def minimize(
        loss_func_wrt_paramvec_and_step=None,
        grad_func_wrt_paramvec_and_step=None,
        save_func_wrt_param_dict=None,
        callback_func_wrt_param_dict=None,
        callback_kwargs=None,
        param_tfm_manager=None,
        dim_P=None,
        init_param_dict=None,
        n_line_search_steps=10,
        n_terms_approx_hessian=10,
        **kwargs):
    """ Minimize provided loss function using L-BFGS algorithm

    Returns
    -------
    param_dict : dict
        Contains estimated parameters that minimize the loss
    alg_state_dict : dict
        Contains algorithm information (num steps completed, etc.)
    """
    pprint('[scipy_lbfgs_minimizer] Begin training...')
    pprint('--n_line_search_steps  %.3f' % n_line_search_steps)
    pprint('--n_terms_approx_hessian %.3f' % n_terms_approx_hessian)

    # Parse user input
    n_line_search_steps = int(n_line_search_steps)
    n_terms_approx_hessian = int(n_terms_approx_hessian)

    # Convert provided common param dict
    # to a flat 1D array with unconstrained values
    param_vec = param_tfm_manager.flatten_to_differentiable_param_vec(
        init_param_dict,
        **dim_P)

    # Warmup
    start_time_sec = time.time()
    init_loss_val = loss_func_wrt_paramvec_and_step(param_vec, step_id=0)
    loss_eval_time_sec = time.time() - start_time_sec
    pprint("Loss     @ init: %8.3f sec | val %.6e" % (
        loss_eval_time_sec, init_loss_val))
    pprint("Params   @ init: %8s     | %5d params | l2 norm / entry %.4e" % (
        ' ',
        param_vec.size,
        calc_l2_norm_of_vector_per_entry(param_vec)))
    start_time_sec = time.time()
    init_grad_vec = grad_func_wrt_paramvec_and_step(param_vec, step_id=0)
    elapsed_time_sec = time.time() - start_time_sec
    init_grad_norm_per_entry = calc_l2_norm_of_vector_per_entry(init_grad_vec)
    pprint("Gradient @ init: %8.3f sec | %5d params | l2 norm / entry %.4e" % (
        elapsed_time_sec, init_grad_vec.size, init_grad_norm_per_entry))

    # Create settings that track algorithm state
    # cur_step, cur_lap, n_laps, n_steps, etc.
    alg_state_kwargs = init_alg_state_kwargs(
        cur_step=0.0,
        **kwargs)
    n_steps = alg_state_kwargs['n_steps']
    if 'output_path' in alg_state_kwargs:
        laps_to_save_str, steps_to_save_str = calc_laps_when_snapshots_saved(
            return_str=True,
            keep_first=5,
            keep_last=5,
            **alg_state_kwargs)
        pprint("Snapshots will be saved at intervals:")
        pprint("   laps: %s" % laps_to_save_str)
        pprint("  steps: %s" % steps_to_save_str)
        pprint("Snapshot saved to --output_path:\n%s" % (
            alg_state_kwargs['output_path']))

    # Translate settings into scipy's specific options format
    options_dict = dict(
        maxiter=n_steps,
        maxfun=n_line_search_steps * n_steps,
        maxcor=n_terms_approx_hessian,
        maxls=n_line_search_steps,
        ftol=0.0,
        gtol=0.0,
        )
    alg_state_kwargs['cur_loss_val'] = init_loss_val

    ## Define special callback function
    # Which does things like print progress at relevant steps
    # Save snapshots to files at relevant steps, etc.
    def my_callback_func(
            cur_param_vec,
            is_init=False,
            alg_state_kwargs=alg_state_kwargs):  
        # Update step counter, timer, etc.
        if not is_init:
            alg_state_kwargs.update(
                update_alg_state_kwargs(
                    **alg_state_kwargs))
        if do_print_now(**alg_state_kwargs) or do_save_now(**alg_state_kwargs):            
            cur_loss_val = loss_func_wrt_paramvec_and_step(cur_param_vec)
            alg_state_kwargs['cur_loss_val'] = cur_loss_val

        if do_print_now(**alg_state_kwargs):            
            pprint(make_status_string(
                **alg_state_kwargs)) # assume cur_loss_val is inside
            save_status_to_txt_files(
                **alg_state_kwargs)
            alg_state_kwargs.update(
                update_alg_state_kwargs_after_print(**alg_state_kwargs))

        if do_save_now(**alg_state_kwargs):
            param_dict = param_tfm_manager.unflatten_to_common_param_dict(
                cur_param_vec, **dim_P)
            if save_func_wrt_param_dict is not None:
                save_func_wrt_param_dict(
                    param_dict=param_dict,
                    **alg_state_kwargs)
            if callback_func_wrt_param_dict is not None:
                callback_func_wrt_param_dict(
                    param_dict=param_dict,
                    losstrain_ttl=alg_state_kwargs.get('cur_loss_val', init_loss_val),
                    alg_state_kwargs=alg_state_kwargs,
                    **callback_kwargs)
            alg_state_kwargs.update(
                update_alg_state_kwargs_after_save(**alg_state_kwargs))

    ## Run training ...
    my_callback_func(param_vec, is_init=True)
    if n_steps > 0:
        opt_result_obj = scipy.optimize.minimize(
            loss_func_wrt_paramvec_and_step,
            param_vec,
            method='l-bfgs-b',
            jac=grad_func_wrt_paramvec_and_step,
            options=options_dict,
            callback=my_callback_func)
        pprint('[scipy_lbfgs_minimizer] msg %s' % opt_result_obj.message)
        param_vec = opt_result_obj.x
        # Relies on alg_state_kwargs already being defined in callback
        my_callback_func(param_vec)

    param_dict = param_tfm_manager.unflatten_to_common_param_dict(
        param_vec, **dim_P)
    pprint('[scipy_lbfgs_minimizer] Done with training.')
    return param_dict, alg_state_kwargs    
