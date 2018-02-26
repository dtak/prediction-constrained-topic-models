import numpy as np

def make_percentile_str(
        arr,
        percentiles=[0, 1, 10, 50, 90, 99, 100],
        fmt_str="%4d",
        sep_str='  '):
    msg_list = list()
    for p in percentiles:
        cur_fmt = "%3d%%:" + fmt_str 
        msg_list.append(
            cur_fmt % (p, np.percentile(arr, p)))
    return sep_str.join(msg_list)
