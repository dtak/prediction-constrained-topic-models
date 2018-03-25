import os
import sys
import numpy as np
import time
import psutil
from sklearn.externals import joblib

default_settings_alg_io = None

def is_lap_in_custom_str(
        cur_lap=None,
        laps_to_save_custom='',
        ):
    ''' Determine if current lap is specified by user custom lap list

    Returns
    -------
    is_custom : boolean

    Examples
    --------
    >>> is_lap_in_custom_str(1.2, '1,1.5,3.00')
    False
    >>> is_lap_in_custom_str(1.5, '1,1.5,3.00')
    True
    >>> is_lap_in_custom_str(3, '1,1.5,3.00')
    True
    >>> is_lap_in_custom_str(3.0001, '1,1.5,3.00')
    False
    >>> is_lap_in_custom_str(1.2, '')
    False
    >>> is_lap_in_custom_str(1.2, 'abc')
    Traceback (most recent call last):
     ...
    ValueError: could not convert string to float: abc
    '''
    list_of_laps_with_no_empties = map(
        float, filter(None, str(laps_to_save_custom).split(',')))
    if len(list_of_laps_with_no_empties) > 0:
        # Do in numerically robust way
        is_match_M = np.abs(
            np.asarray(list_of_laps_with_no_empties) - float(cur_lap))
        is_custom = np.min(is_match_M) < 1e-7
    else:
        is_custom = False
    return is_custom

def do_print_now(
        cur_step=0,
        cur_lap=0,
        n_steps=0,
        n_steps_to_print_early=5,
        n_steps_between_print=None,
        n_seconds_between_print=None,
        laps_to_save_custom='',
        elapsed_time_sec=0.0,
        last_print_step=0,
        last_print_sec=0.0,
        step_offset=0,
        is_converged=False,
        **kwargs):
    corrected_step = cur_step - step_offset
    is_early = corrected_step <= n_steps_to_print_early
    is_last = corrected_step == n_steps
    is_custom = is_lap_in_custom_str(cur_lap, laps_to_save_custom)
    do_print = is_early or is_last or is_custom
    if n_steps_between_print is not None and n_steps_between_print > 0:
        do_print = do_print or \
            (corrected_step % int(n_steps_between_print) == 0)
    if n_seconds_between_print is not None and n_seconds_between_print > 0:
        do_print = do_print or \
            (elapsed_time_sec - last_print_sec > n_seconds_between_print)
    if not do_print and is_converged:
        do_print = True
    return do_print

def do_save_now(
        cur_step=0,
        cur_lap=0,
        n_steps=0,
        n_steps_to_save_early=5,
        n_steps_between_save=None,
        n_seconds_between_save=None,
        laps_to_save_custom='',
        elapsed_time_sec=0.0,
        last_save_sec=0.0,
        last_save_step=0,
        step_offset=0,
        is_converged=False,
        **kwargs):
    corrected_step = cur_step - step_offset
    is_early = corrected_step <= n_steps_to_save_early
    is_last = corrected_step == n_steps
    is_custom = is_lap_in_custom_str(cur_lap, laps_to_save_custom)
    do_save = is_early or is_last or is_custom
    if n_steps_between_save is not None and n_steps_between_save > 0:
        do_save = do_save or \
            ((corrected_step) % int(n_steps_between_save) == 0)
    if n_seconds_between_save is not None and n_seconds_between_save > 0:
        do_save = do_save or \
            (elapsed_time_sec - last_save_sec > n_seconds_between_save)
    if not do_save and is_converged:
        do_save = True
    return do_save

def cast_or_None(val, type):
    ''' Convert provided value to spec'd type, or return None

    Examples
    --------
    >>> cast_or_None(None, int)
    
    >>> cast_or_None(3, int)
    3
    >>> cast_or_None('33', int)
    33
    >>> cast_or_None('12', float)
    12.0
    '''
    if val is not None:
        return type(val)
    return None

def init_alg_state_kwargs(
        cur_step=0,
        n_laps=0,
        n_batches=1,
        output_path=None,
        step_offset=0,
        n_seconds_between_save=-1,
        n_seconds_between_print=-1,
        n_steps_between_save=None,
        n_steps_between_print=None,
        n_steps_to_save_early=0,
        n_steps_to_print_early=0,
        laps_to_save_custom='',
        param_output_fmt='dump',
        **kwargs):
    n_laps = float(n_laps)
    n_batches = int(n_batches)
    n_steps = int(n_laps * n_batches)

    start_time_sec = time.time()
    elapsed_time_sec = 0.0
    cur_mem = getMemUsageOfCurProcess_MiB('rss')
    cur_swp = getMemUsageOfCurProcess_MiB('vms')
    return dict(
        cur_step=cur_step,
        step_offset=int(step_offset),
        cur_lap=cur_step / float(n_batches),
        cur_mem_MiB=cur_mem,
        cur_swp_MiB=cur_swp,
        max_mem_MiB=cur_mem,
        max_swp_MiB=cur_swp,
        n_steps=n_steps,
        n_laps=n_laps,
        n_batches=n_batches,
        start_time_sec=start_time_sec,
        elapsed_time_sec=elapsed_time_sec,
        output_path=output_path,
        param_output_fmt=param_output_fmt,
        laps_to_save_custom=str(laps_to_save_custom),
        n_steps_between_save=cast_or_None(n_steps_between_save, int),
        n_steps_between_print=cast_or_None(n_steps_between_print, int),
        n_seconds_between_save=cast_or_None(n_seconds_between_save, float),
        n_seconds_between_print=cast_or_None(n_seconds_between_print, float),
        n_steps_to_save_early=cast_or_None(n_steps_to_save_early, int),
        n_steps_to_print_early=cast_or_None(n_steps_to_print_early, int))

def update_alg_state_kwargs(
        **alg_state_kwargs):
    alg_state_kwargs['cur_step'] += 1
    alg_state_kwargs['cur_lap'] = \
        alg_state_kwargs['cur_step'] / float(alg_state_kwargs['n_batches'])
    alg_state_kwargs['elapsed_time_sec'] = \
        time.time() - alg_state_kwargs['start_time_sec']
    alg_state_kwargs['cur_mem_MiB'] = getMemUsageOfCurProcess_MiB('rss')
    alg_state_kwargs['cur_swp_MiB'] = getMemUsageOfCurProcess_MiB('vms')
    alg_state_kwargs['max_mem_MiB'] = np.maximum(
        alg_state_kwargs['cur_mem_MiB'],
        alg_state_kwargs['max_mem_MiB'])
    alg_state_kwargs['max_swp_MiB'] = np.maximum(
        alg_state_kwargs['cur_swp_MiB'],
        alg_state_kwargs['max_swp_MiB'])
    return alg_state_kwargs

def update_alg_state_kwargs_after_print(
        **alg_state_kwargs):
    alg_state_kwargs['last_print_step'] = alg_state_kwargs['cur_step']
    alg_state_kwargs['last_print_sec'] = alg_state_kwargs['elapsed_time_sec']
    return alg_state_kwargs

def update_alg_state_kwargs_after_save(
        **alg_state_kwargs):
    alg_state_kwargs['last_save_step'] = alg_state_kwargs['cur_step']
    alg_state_kwargs['last_save_sec'] = alg_state_kwargs['elapsed_time_sec']
    return alg_state_kwargs


def save_status_to_txt_files(
        cur_lap=0.0,
        cur_step=0.0,
        cur_loss_val=None,
        cur_grad_norm_per_entry=None,
        cur_step_size=None,
        cur_loss_x=None,
        cur_loss_y=None,
        cur_mem_MiB=None,
        cur_swp_MiB=None,
        elapsed_time_sec=0.0,
        output_path=None,
        **kwargs):
    for key in [
            'min_step_size_L1',
            'median_step_size_L1',
            'max_step_size_L1',
            'sum_step_size_L1']:
        if key not in kwargs:
            continue
        append_to_txtfile(
            output_path=output_path,
            fmt='%.5e',
            **{(key):kwargs[key]})
    append_to_txtfile(
        lap=cur_lap, fmt='%.5f', output_path=output_path)
    append_to_txtfile(
        step=cur_step, fmt='%d', output_path=output_path)
    append_to_txtfile(
        elapsed_time_sec=elapsed_time_sec,
        fmt='%.6e', output_path=output_path)
    if cur_mem_MiB is not None:
        append_to_txtfile(
            cur_mem_MiB=cur_mem_MiB, fmt='%.3f', output_path=output_path)
    if cur_swp_MiB is not None:
        append_to_txtfile(
            cur_swp_MiB=cur_swp_MiB, fmt='%.3f', output_path=output_path)
    if cur_step_size is not None:
        append_to_txtfile(
            step_size=cur_step_size, fmt='%.6f', output_path=output_path)
    if cur_loss_val is not None:
        append_to_txtfile(
            loss=cur_loss_val, fmt='%.6e', output_path=output_path)
    if cur_grad_norm_per_entry is not None:
        append_to_txtfile(
            grad_l2_norm_per_entry=cur_grad_norm_per_entry,
            fmt='%.6e', output_path=output_path)
    if cur_loss_x is not None:
        append_to_txtfile(
            loss=cur_loss_x, fmt='%.6e', output_path=output_path)
    if cur_loss_y is not None:
        append_to_txtfile(
            loss=cur_loss_y, fmt='%.6e', output_path=output_path)

def array2string(arr):
    return np.array2string(
        arr,
        precision=3,
        )

def make_status_string(
        cur_step=0,
        n_laps=0,
        n_batches=1,
        cur_param_str=None,
        cur_loss_val=None,
        cur_grad_norm_per_entry=None,
        cur_loss_x=None,
        cur_loss_y=None,
        cur_mem_MiB=None,
        cur_swp_MiB=None,
        cur_is_feasible=None,
        elapsed_time_sec=0.0,
        **kwargs):
    '''
    '''
    msg_str = "step %6d/%d   lap %8.3f/%d  after %8.2f sec" % (
        cur_step,
        int(n_laps * n_batches),
        cur_step / float(n_batches),
        int(n_laps),
        elapsed_time_sec,
        )
    if cur_mem_MiB is not None:
        msg_str += "  rss %6.2f MiB" % (cur_mem_MiB)
    if cur_swp_MiB is not None:
        msg_str += "  swp %6.2f MiB" % (cur_swp_MiB)

    if cur_loss_val is not None:
        msg_str += "  loss % .6e" % (cur_loss_val)
    if cur_grad_norm_per_entry is not None:
        msg_str += "  |g|  % .3e" % (cur_grad_norm_per_entry)
    if cur_loss_x is not None:
        msg_str += "  x_loss % .6e" % (cur_loss_x)
    if cur_loss_y is not None:
        msg_str += "  y_loss % .6e" % (cur_loss_y)
    if cur_is_feasible is not None:
        msg_str += " is_feas %d" % (cur_is_feasible)
    if isinstance(cur_param_str, str):
        msg_str += " " + cur_param_str
    return msg_str


def getMemUsageOfCurProcess_MiB(field='rss'):
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = getattr(process.memory_info(), field)
    mem_MiB = mem / float(2 ** 20)
    return mem_MiB


def append_to_txtfile(
        output_path='/tmp/',
        fmt='%.3e',
        prefix='trace',
        **kwargs):
    if output_path is None:
        return
    for key, val in kwargs.items():
        txt_path = os.path.join(output_path, prefix + '_' + key + '.txt')
        with open(txt_path, 'a') as f:
            f.write(fmt % val + '\n')

def update_symbolic_link(hardfile, linkfile):
    if linkfile.endswith(os.path.sep):
        linkfile = linkfile[:-len(os.path.sep)]
    if os.path.islink(linkfile):
        os.unlink(linkfile)
    if os.path.exists(linkfile):
        os.remove(linkfile)
    if os.path.exists(hardfile):
        # TODO: Handle Windows os case
        os.symlink(hardfile, linkfile)

def calc_laps_when_snapshots_saved(
        n_batches=1,
        n_laps=1,
        return_str=False,
        n_keep_left=5,
        n_keep_right=5,
        **alg_state_kwargs):
    alg_state_kws = dict(**alg_state_kwargs)
    alg_state_kws['cur_step'] = 0
    alg_state_kws['n_batches'] = int(n_batches)
    alg_state_kws['n_laps'] = float(n_laps)
    alg_state_kws = init_alg_state_kwargs(**alg_state_kws)
    n_steps = alg_state_kws['n_steps']
    do_save_laps = list()
    do_save_steps = list()
    for step_id in range(0, n_steps + 1):
        if do_save_now(**alg_state_kws):
            do_save_laps.append(alg_state_kws['cur_lap'])
            do_save_steps.append(alg_state_kws['cur_step'])
        alg_state_kws = update_alg_state_kwargs(**alg_state_kws)
    if n_keep_left > 0 and n_keep_right > 0:
        do_save_laps = np.unique(np.hstack([
            do_save_laps[:n_keep_left],
            do_save_laps[-n_keep_right:]])).tolist()
        do_save_steps = np.unique(np.hstack([
            do_save_steps[:n_keep_left],
            do_save_steps[-n_keep_right:]])).tolist()
    if return_str:
        steps_str = ','.join(['%6d' % a for a in do_save_steps])
        laps_str  = ','.join(['%6.3g' % a for a in do_save_laps])
        return laps_str, steps_str
    else:
        return do_save_laps, do_save_steps

if __name__ == '__main__':
    do_save_laps = calc_laps_when_snapshots_saved(
        n_batches=1,
        n_laps=20,
        n_steps_between_save=5,
        laps_to_save_custom='1,3,5,7,11,13,17,19')
    print do_save_laps