'''
Define invertible transform from unit interval to real line.

Examples
--------
>>> from autograd import elementwise_grad
>>> g_auto = elementwise_grad(
...     _log_logistic_sigmoid_not_vectorized)
>>> g_manual = elementwise_grad(log_logistic_sigmoid)
>>> vals = np.linspace(-5000., 5000., 100)
>>> for x in vals: assert np.allclose(g_auto(x), g_manual(x))

# Can successfully call g_manual on array of values
>>> np.all(np.isfinite(g_manual(vals)))
True

# Cannot do so with autograd
>>> g_auto(vals)
Traceback (most recent call last):
...
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
'''

import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from autograd.core import primitive
from autograd import elementwise_grad, grad
try: 
    from autograd.extend import primitive, defvjp  # defvjp is now a function
except ImportError:
    from autograd.core import primitive
    defvjp = None

def to_common_arr(x):
    return log_logistic_sigmoid(x)

def _log_logistic_sigmoid(x_real):
    ''' Compute log of logistic sigmoid transform from real line to unit interval.

    Numerically stable and fully vectorized.

    Args
    ----
    x_real : array-like, with values in (-infty, +infty)

    Returns
    -------
    log_p_real : array-like, size of x_real, with values in <= 0
    '''
    if not isinstance(x_real, float):
        out = np.zeros_like(x_real)
        mask1 = x_real > 50.0
        out[mask1] = - np.log1p(np.exp(-x_real[mask1]))
        mask0 = np.logical_not(mask1)
        out[mask0] = x_real[mask0]
        out[mask0] -= np.log1p(np.exp(x_real[mask0]))
        return out
    return _log_logistic_sigmoid_not_vectorized(x_real)

def _log_logistic_sigmoid_not_vectorized(x_real):
    if x_real > 50.0:
        return - np.log1p(np.exp(-x_real))
    else:
        return x_real - np.log1p(np.exp(x_real))

@primitive
def log_logistic_sigmoid(x):
    return _log_logistic_sigmoid(x)

# Definite gradient function via manual formula
# Supporting different versions of autograd software
if defvjp is not None:
    # Latest version of autograd
    def _vjp__log_logistic_sigmoid(ans, x):
        def _my_gradient(g, x=x, ans=ans):
            x = np.asarray(x)
            return np.full(x.shape, g) * (1 - np.exp(ans))
        return _my_gradient
    defvjp(
        log_logistic_sigmoid,
        _vjp__log_logistic_sigmoid,
        )
elif hasattr(primitive, 'defvjp'):
    # Slightly older version of autograd
    def _vjp__log_logistic_sigmoid(g, ans, vs, gvs, x):
        x = np.asarray(x)
        return np.full(x.shape, g) * (1 - np.exp(ans))
    log_logistic_sigmoid.defvjp(_vjp__log_logistic_sigmoid)
else:
    # Older version of autograd
    def _make_grad_product(ans, x):
        x = np.asarray(x)
        def grad_product(g):
            return np.full(x.shape, g) * (1 - np.exp(ans))
        return grad_product
    log_logistic_sigmoid.defgrad(_make_grad_product)
