from pprint_logging import pprint, config_pprint_logging
from util_pprint_percentiles import make_percentile_str
from util_io_topic_snapshot import (
    load_param_dict_from_topic_model_snapshot,
    save_topic_model_snapshot)

from util_io_training import (
    do_print_now,
    do_save_now,
    default_settings_alg_io,
    init_alg_state_kwargs,
    update_alg_state_kwargs,
    make_status_string,
    save_params_to_disk_and_callback,
    save_status_to_txt_files,
    append_to_txtfile,
    update_alg_state_kwargs_after_print,
    update_alg_state_kwargs_after_save,
    update_symbolic_link,
    )

from util_setup import (
    setup_detect_taskid_and_insert_into_output_path,
    setup_random_seed,
    setup_output_path)

from util_io_csr import (
    load_csr_matrix,
    save_csr_matrix)

from util_io_txt import (
    load_list_of_strings_from_txt,
    load_list_of_unicode_from_txt,
    )

from util_array import (
    toCArray,
    as1D,
    as2D,
    as3D)
