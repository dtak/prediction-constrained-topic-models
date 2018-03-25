import time
import numpy as np

def start_timer_segment(etime_info_dict, label):
    etime_info_dict[label + "_start"] = time.time()
    return etime_info_dict

def stop_timer_segment(etime_info_dict, label):
    try:
        etime_info_dict[label + "_elapsed"] = \
            time.time() - etime_info_dict[label + "_start"]
    except KeyError as e:
        pass
    return etime_info_dict

def pprint_timer_segments(
        etime_info_dict, total_key='total', prefix=''):
    line_list = list()
    total_elapsed_key= total_key + "_elapsed"
    try:
        total_elapsed = etime_info_dict[total_elapsed_key]
    except KeyError:
        # Find earliest start time and mark that as elapsed
        earliest_start_time = np.inf
        for key in etime_info_dict:
            if key.endswith('_start'):
                etime = etime_info_dict[key]
                if etime < earliest_start_time:
                    earliest_start_time = etime
        total_elapsed = time.time() - earliest_start_time
    total_measured = 0.0
    for key in etime_info_dict:
        if key.endswith("_elapsed"):
            etime = etime_info_dict[key]
            total_measured += etime
            msg_line = "%-10s %5.1f%% %8.2f sec %s" % (
                prefix,
                float(etime / total_elapsed) * 100,
                etime,
                key.replace('_elapsed', ''))
            line_list.append(msg_line)

    total_other = total_elapsed - total_measured
    msg_line = "%-10s %5.1f%% %8.2f sec %s" % (
        prefix,
        float(total_other / total_elapsed) * 100,
        total_other,
        'other_unmeasured')
    line_list.append(msg_line)

    return '\n'.join(line_list) + '\n'
