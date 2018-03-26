try:
    from calc_nef_map_pi_d_K import (
        calc_nef_map_pi_d_K__tensorflow,
        _calc_nef_map_pi_d_K__tensorflow_graph,
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    _calc_nef_map_pi_d_K__tensorflow_graph = None

from calc_nef_map_pi_d_K import (
    calc_nef_map_pi_d_K,
    make_convex_alpha_minus_1,
    DefaultDocTopicOptKwargs,
    calc_nef_map_pi_d_K__autograd,
    calc_nef_map_pi_d_K__cython,
    )