import os
import sys
import numpy as np

from collections import OrderedDict
from sscape.utils_io import load_param_dict_from_topic_model_snapshot

def load_param_dict_at_specific_snapshot(
        snapshot_path=None,
        task_path=None,
        lap=None,
        download_if_necessary=True,
        rsync_path=None,
        local_path=None,
        remote_path=None,
        snapshot_suffix='topic_model_snapshot/',
        w_txt_basename="w_CK.txt",
        add_bias_term_to_w_CK=0.0,
        **kwargs):
    if snapshot_path is None:
        snapshot_path = make_snapshot_path_for_lap(
            task_path, lap=lap, snapshot_suffix=snapshot_suffix)

    if not os.path.exists(snapshot_path) and download_if_necessary:
        download_snapshot(snapshot_path, rsync_path, local_path, remote_path)

    GP = load_param_dict_from_topic_model_snapshot(
        snapshot_path=snapshot_path,
        w_txt_basename=w_txt_basename,
        add_bias_term_to_w_CK=add_bias_term_to_w_CK,
        )
    return GP


def download_snapshot(
        snapshot_path=None,
        rsync_path=None,
        local_path=None,
        remote_path=None,
        ):
    if local_path is None:
        try:
            local_path = os.environ['XHOST_LOCAL_PATH']
        except KeyError:
            raise ValueError("Bad value for local_path: %s" % local_path)
    if remote_path is None:
        try:
            remote_path = os.environ['XHOST_REMOTE_PATH']
        except KeyError:
            raise ValueError("Bad value for remote_path: %s" % remote_path)
    if rsync_path is None:
        try:
            rsync_path = os.environ['XHOST_RSYNC_PATH']
        except KeyError:
            raise ValueError("Bad value for rsync_path: %s" % rsync_path)

    local_path = os.path.abspath(local_path)
    remote_path = os.path.abspath(remote_path)

    if snapshot_path.count(local_path):
        local_snapshot_path = snapshot_path
        remote_snapshot_path = snapshot_path.replace(local_path, remote_path)
    elif snapshot_path.count(remote_path):
        remote_snapshot_path = snapshot_path
        local_snapshot_path = snapshot_path.replace(remote_path, local_path)

    cmd_str = "bash rsync_specific_snapshot.sh %s" % remote_snapshot_path
    print "ATTEMPTING DOWNLOAD via RSYNC..."
    print "CMD:\n", cmd_str
    old_path = os.getcwd()
    os.chdir(rsync_path)
    print os.getcwd()
    ans = os.system(cmd_str)
    os.chdir(old_path)
    if int(str(ans)) != 0:
        raise ValueError("BAD DOWNLOAD: ANSWER CODE %s" % ans)
    return True

def make_snapshot_path_for_lap(
        task_path=None, lap=None, snapshot_suffix='topic_model_snapshot/'):
    if isinstance(lap, float) or isinstance(lap, int):
        best_lap = float(lap)
    else:
        raise ValueError("Bad value for lap %s" % lap)
    snapshot_path = os.path.join(
        task_path,
        'lap%011.3f_%s' % (best_lap, snapshot_suffix))
    return snapshot_path
