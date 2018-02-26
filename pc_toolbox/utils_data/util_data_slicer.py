import numpy as np

def make_slice_for_step(
        step_id=0,
        n_total=0,
        n_batches=1,
        seed=42,
        **kwargs):
    ''' Compute slice for provided step

    If step_id < 0, always given the first slice
    Otherwise, give a random slice

    Returns
    -------
    cur_slice : slice object
    '''
    if step_id >= 0:
        ## Seed the random generator with current lap number
        prng = np.random.RandomState(seed + (step_id // n_batches))
        batch_order = prng.permutation(n_batches)
        batch_id = batch_order[step_id % n_batches]
    else:
        batch_id = 0
    batch_size = int(np.ceil(n_total / float(n_batches)))
    start = batch_id * batch_size
    stop = np.minimum(n_total, (batch_id + 1) * batch_size)
    return slice(start, stop)


