import os

## Create defaults on load
## Overriding some options if set in os.environ namespace
def make_default_kwargs():
    lstep_kwargs = dict(
        pi_max_iters=100,
        pi_min_iters=10,
        pi_converge_thr=0.0001,
        pi_step_size=0.005,
        pi_max_step_size=0.1,
        pi_min_step_size=1.0e-9,
        pi_step_decay_rate=0.75,
        pi_min_mass_preserved_to_trust_step=0.25)
    for key, val in lstep_kwargs.items():
        if key in os.environ:
            if isinstance(val, float):
                lstep_kwargs[key] = float(os.environ[key])
            else:
                lstep_kwargs[key] = int(os.environ[key])
            print ">>> OVERRIDE DEFAULT LSTEP KW: %s = %s" % (
                key, os.environ[key])
    return lstep_kwargs
DefaultDocTopicOptKwargs = make_default_kwargs()
