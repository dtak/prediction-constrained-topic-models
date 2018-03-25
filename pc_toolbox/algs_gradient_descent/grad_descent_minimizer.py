import argparse
import numpy as np
import time

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

def calc_l2_norm_of_vector_per_entry(grad_vec):
    return np.sqrt(np.sum(np.square(grad_vec))) / float(grad_vec.size)

def minimize(
        loss_func_wrt_paramvec_and_step=None,
        grad_func_wrt_paramvec_and_step=None,
        save_func_wrt_param_dict=None,
        callback_func_wrt_param_dict=None,
        callback_kwargs=None,
        param_tfm_manager=None,
        dim_P=None,
        init_param_dict=None,
        step_direction='steepest',
        step_size=0.01,
        decay_rate=1.0,
        decay_interval=25,
        decay_staircase=0,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        max_l2_norm_of_grad_per_entry=10.0,
        **kwargs):
    pprint('[grad_descent_minimizer] Begin training...')
    pprint('--step_direction %s' % step_direction)
    pprint('--step_size  %.3f' % step_size)
    pprint('--decay_rate %.3f' % decay_rate)

    # Parse user input
    step_direction = str(step_direction)
    assert step_direction in ['adam', 'steepest']
    step_size = float(step_size)
    decay_rate = float(decay_rate)
    decay_staircase = int(decay_staircase)
    decay_interval = float(decay_interval)
    b1 = float(b1)
    b2 = float(b2)
    eps = float(eps)

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

    # Adam estimates of gradient mean/variance
    m = np.zeros_like(param_vec)
    v = np.zeros_like(param_vec)

    cur_step_size = step_size
    cur_loss_val = init_loss_val
    cur_grad_norm_per_entry = init_grad_norm_per_entry
    for step_id in xrange(0, n_steps + 1):
        if step_id > 0:
            grad_vec = grad_func_wrt_paramvec_and_step(param_vec, step_id=step_id)

            cur_grad_norm_per_entry = calc_l2_norm_of_vector_per_entry(grad_vec)
            assert np.isfinite(cur_grad_norm_per_entry)
            if cur_grad_norm_per_entry > max_l2_norm_of_grad_per_entry:
                warn_msg = (
                    'WARNING: clipping gradient enforced.'
                    + '\n cur l2 norm / entry = %.2e'
                    + '\n new l2 norm / entry = %.2e')
                pprint(warn_msg % (
                    cur_grad_norm_per_entry,
                    max_l2_norm_of_grad_per_entry))
                grad_vec *= max_l2_norm_of_grad_per_entry / cur_grad_norm_per_entry
                cur_grad_norm_per_entry = calc_l2_norm_of_vector_per_entry(grad_vec)
            assert cur_grad_norm_per_entry <= max_l2_norm_of_grad_per_entry

            # Decay learning rate, like tensorflow's exponential decay
            if decay_staircase:
                cur_step_count = int(step_id) // int(decay_interval)
            else:
                cur_step_count = float(step_id) / float(decay_interval)
            cur_step_size = step_size * decay_rate ** (cur_step_count)

            if step_direction == 'adam':
                g = grad_vec
                m = (1 - b1) * g      + b1 * m  # First  moment estimate.
                v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
                mhat = m / (1 - b1**(step_id))    # Bias correction.
                vhat = v / (1 - b2**(step_id))
                step_vec = -1.0 * cur_step_size * mhat / (np.sqrt(vhat) + eps)
            elif step_direction.count('steep'):
                step_vec = -1.0 * cur_step_size * grad_vec
            else:
                raise ValueError("Unrecognized step_direction: %s" % step_direction)
            param_vec = param_vec + step_vec
            assert np.all(np.isfinite(param_vec))

            # Update step counter, timer, etc.
            alg_state_kwargs = update_alg_state_kwargs(
                **alg_state_kwargs)

        if do_print_now(**alg_state_kwargs):
            cur_loss_val = loss_func_wrt_paramvec_and_step(param_vec, step_id=step_id)
            pprint(make_status_string(
                cur_loss_val=cur_loss_val,
                cur_grad_norm_per_entry=cur_grad_norm_per_entry,
                **alg_state_kwargs))
            save_status_to_txt_files(
                cur_loss_val=cur_loss_val,
                cur_grad_norm_per_entry=cur_grad_norm_per_entry,
                cur_step_size=cur_step_size,
                **alg_state_kwargs)
            alg_state_kwargs = update_alg_state_kwargs_after_print(
                **alg_state_kwargs)

        if do_save_now(**alg_state_kwargs):
            param_dict = param_tfm_manager.unflatten_to_common_param_dict(
                param_vec, **dim_P)
            if save_func_wrt_param_dict is not None:
                save_func_wrt_param_dict(
                    param_dict=param_dict,
                    **alg_state_kwargs)
            if callback_func_wrt_param_dict is not None:
                callback_func_wrt_param_dict(
                    param_dict=param_dict,
                    losstrain_ttl=cur_loss_val,
                    alg_state_kwargs=alg_state_kwargs,
                    **callback_kwargs)
            alg_state_kwargs = update_alg_state_kwargs_after_save(
                **alg_state_kwargs)

    param_dict = param_tfm_manager.unflatten_to_common_param_dict(
        param_vec, **dim_P)
    pprint('[grad_descent_minimizer] Done with training.')
    return param_dict, alg_state_kwargs    
