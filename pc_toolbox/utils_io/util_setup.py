import os
import sys
import subprocess
import shutil
import hashlib
import numexpr as ne
import glob
from distutils.dir_util import mkpath

from pprint_logging import config_pprint_logging, pprint

def setup_detect_taskid_and_insert_into_output_path(
        output_path=None,
        do_only_run_if_empty=0,
        return_extras=True,
        **kwargs):
    stdout_file = None
    stderr_file = None

    output_path = os.path.abspath(output_path).rstrip(os.path.sep)
    output_folders = output_path.split(os.path.sep)
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        job_id_str = os.environ['SLURM_ARRAY_JOB_ID']
        taskidstr = os.environ['SLURM_ARRAY_TASK_ID']
        output_folders[-1] = taskidstr
        stdout_file = os.path.join(
            os.environ['XHOST_LOG_DIR'],
            '%s.%s.out' % (job_id_str, taskidstr))
        stderr_file = stdout_file.replace('.out', '.err')

    elif 'LSB_JOBINDEX' in os.environ:
        job_id_str = os.environ['LSB_JOBID']
        taskidstr = os.environ['LSB_JOBINDEX']
        output_folders[-1] = taskidstr 
        if 'LSB_ERRORFILE' in os.environ:
            stderr_file = os.environ['LSB_ERRORFILE']
            stdout_file = stderr_file.replace('.err', '.out')

    elif 'SGE_TASK_ID' in os.environ:
        job_id_str = os.environ['JOB_ID']
        taskidstr = os.environ['SGE_TASK_ID']
        output_folders[-1] = taskidstr

        stdout_file = os.environ['SGE_STDOUT_PATH']
        stderr_file = os.environ['SGE_STDERR_PATH']
    elif 'XHOST_TASK_ID' in os.environ:
        job_id_str = 'none'
        taskidstr = os.environ['XHOST_TASK_ID']
        output_folders[-1] = taskidstr
    else:
        job_id_str = 'none'
    taskidstr = output_folders[-1]
    output_path = os.path.sep.join(output_folders)    

    do_only_run_if_empty = int(do_only_run_if_empty)
    print 'do_only_run_if_empty', do_only_run_if_empty
    if do_only_run_if_empty != 0:
        snap_list = glob.glob(os.path.join(output_path, 'lap*_snapshot'))
        print 'len(snap_list)', len(snap_list)
        if do_only_run_if_empty > 0 and len(snap_list) >= 10:
            raise ValueError("RUN ALREADY COMPLETED. SKIPPING...")
        elif do_only_run_if_empty < 0 and len(snap_list) < 10:
            raise ValueError("RUN NOT YET COMPLETED. SKIPPING...")

    if return_extras:
        return output_path, job_id_str, taskidstr, stdout_file, stderr_file
    else:
        return output_path

def setup_output_path(
        output_path=None,
        do_make_new_dir=True,
        **alg_state_kwargs):
    if output_path is None:
        return None
    output_path, job_id_str, taskidstr, stdout_file, stderr_file = \
        setup_detect_taskid_and_insert_into_output_path(
            output_path=output_path,
            return_extras=True, **alg_state_kwargs)
    if do_make_new_dir:
        make_empty_output_path(output_path)
    config_pprint_logging(output_path)

    info_list = list()
    info_list.append('JOB_ID = %s' % job_id_str)
    info_list.append('TASK_ID = %s' % taskidstr)
    uname_list = os.uname()
    info_list.append('hostname = %s' % uname_list[1])
    info_list.append(' '.join(map(str, uname_list)))
    try:
        cpu_list = ne.cpuinfo.cpuinfo.info
        if isinstance(cpu_list, list):
            info_list.append('n_cpus = %d' % len(cpu_list))
            for cpu_info in cpu_list[:4]:
                info_list.append(
                    '%s MHz  %s' % (cpu_info['cpu MHz'], cpu_info['model name']))
    except Exception as e:
        print str(e)
        pass
    info_list.append('')
    with open(os.path.join(output_path, 'machine_info.txt'), 'w') as f:
        for line in info_list:
            pprint(line)
            f.write(line + "\n")  
        if stdout_file is not None:
            f.write(stdout_file + "\n")
            f.write(stderr_file + "\n")

    if stdout_file is not None:
        pprint("making link to stderr_file")
        pprint(stderr_file)
        try:
            os.symlink(stdout_file, os.path.join(output_path, 'grid_log.out'))
            os.symlink(stderr_file, os.path.join(output_path, 'grid_log.err'))
        except Exception as e:
            print str(e)
            pass

    return output_path

def setup_random_seed(
        output_path=None,
        seed=None,
        **kwargs):
    if output_path is None:
        if seed is None:
            return 0
        else:
            return int(seed)
    output_folders = os.path.abspath(output_path).split(os.path.sep)
    # Discard trailing '/' if it exists
    while len(output_folders[-1]) == 0:
        output_folders.pop()
    if seed is None:
        seed_str = str(output_folders[-2])[:5].split('-')[0]
        taskidstr = output_folders[-1]
        seed_str = seed_str[:5] + taskidstr
        seed = int(hashlib.md5(seed_str).hexdigest(), 16) % 10000000
        pprint('seed_str: ' + seed_str)
    pprint('seed: %d' % int(seed))
    return int(seed)


def make_empty_output_path(
        output_path='/tmp/',
        **kwargs):
    ''' Create specified path on the file system with empty contents.

    Postcondition
    -------------
    The directory outputdir will exist, with no content.
    Any required parent paths will be automatically created.
    Any pre-existing content will be deleted, to avoid confusion.
    '''
    if output_path is None:
        return

    # Ensure the path (including all parent paths) exists
    mkpath(output_path)
    # Ensure the path has no content from previous runs
    for the_file in os.listdir(output_path):
        file_path = os.path.join(output_path, the_file)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        if os.path.isdir(file_path):
            if file_path.endswith("topic_model_snapshot"):
                shutil.rmtree(file_path)