from est_local_params__single_doc_map import (
    calc_nef_map_pi_d_K,
    calc_nef_map_pi_d_K__autograd,
    calc_nef_map_pi_d_K__cython,
    DefaultDocTopicOptKwargs,
    )

from est_local_params__many_doc_map import (
    calc_nef_map_pi_DK,
    )

import slda_utils__dataset_manager
import slda_utils__param_io_manager
import slda_utils__param_manager
import slda_utils__init_manager


import slda_loss__autograd
import slda_loss__cython

try:
    import slda_loss__tensorflow
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    slda_loss__tensorflow = None

import slda_snapshot_perf_metrics
