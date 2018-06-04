from est_local_params__single_doc_map import (
    calc_nef_map_pi_d_K,
    calc_nef_map_pi_d_K__autograd,
    calc_nef_map_pi_d_K__cython,
    DefaultDocTopicOptKwargs,
    )

from est_local_params__many_doc_map import (
    calc_nef_map_pi_DK,
    )

from est_local_params__vb_qpiDir_qzCat import (
    calc_elbo_for_many_docs,
    )


import slda_utils__dataset_manager
import slda_utils__param_io_manager
save_topic_model_param_dict = slda_utils__param_io_manager.save_topic_model_param_dict
load_topic_model_param_dict = slda_utils__param_io_manager.load_topic_model_param_dict

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
