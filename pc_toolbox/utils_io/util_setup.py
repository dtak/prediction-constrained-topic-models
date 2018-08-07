import os
import sys
import subprocess
import shutil
import hashlib
import glob
from distutils.dir_util import mkpath
from numexpr.cpuinfo import cpuinfo as numexpr_cpuinfo

from pprint_logging import config_pprint_logging, pprint
import util_watermark

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
    if do_only_run_if_empty != 0:
        pprint("do_only_run_if_empty: %s" % do_only_run_if_empty)
        snap_list = glob.glob(os.path.join(output_path, 'lap*_snapshot'))
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
        cpu_list = numexpr_cpuinfo.info
        if isinstance(cpu_list, list):
            info_list.append('n_cpus = %d' % len(cpu_list))
            for cpu_info in cpu_list[:4]:
                info_list.append(
                    '%s MHz  %s' % (cpu_info['cpu MHz'], cpu_info['model name']))
            if len(cpu_list) > 5:
                info_list.append('...')
            if len(cpu_list) > 4:
                info_list.append(
                    '%s MHz  %s' % (cpu_list[-1]['cpu MHz'], cpu_list[-1]['model name']))

    except Exception as e:
        pprint("Skipping over error in numexpr_cpuinfo call (not necessary, just useful):")
        pprint("    " + str(e))
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
            pprint(str(e))
            pass

    return output_path

def setup_random_seed(
        output_path=None,
        seed=None,
        **kwargs):
    if str(seed).lower() == 'none' or seed == 'from_output_path_and_taskid':
        if output_path is None:
            seed = 8675309
        else:
            output_folders = os.path.abspath(output_path).split(os.path.sep)
            # Discard trailing '/' if it exists
            while len(output_folders[-1]) == 0:
                output_folders.pop()
            seed_str = str(output_folders[-2])[:8].split('-')[0]
            taskidstr = output_folders[-1]
            seed_str = seed_str[:5] + taskidstr
            pprint('[setup_random_seed] seed_str=' + seed_str)
            seed = int(hashlib.md5(seed_str).hexdigest(), 16) % 10000000
    else:
        seed = int(seed)
    pprint('[setup_random_seed] seed=%d' % (seed))
    return seed


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

def write_env_vars_to_txt(
        output_path=None,
        prefixes=(
            ['PC', 'XHOST', 'MKL', 'PYTHON', 'PATH']
            + ['SGE', 'SLURM', 'LSF']
            + ['MPL']),
        ):
    """ Write .txt file to provided output_path with environment var info.

    Post condition
    --------------
    input_environment_vars.txt file written inside output_path.
        Each line contains setting of environment variable. 
    """
    txt_fpath = os.path.join(output_path, 'input_environment_vars.txt')
    with open(txt_fpath, 'w') as f:
        for key in sorted(os.environ):
            for prefix in prefixes:
                if key.startswith(prefix):
                    f.write("%s=%s\n" % (key, os.environ[key]))



def write_user_provided_kwargs_to_txt(
        arg_dict=None,
        output_path=None):
    """ Write .txt file to provided output_path with args info.

    Post condition
    --------------
    input_keyword_args.txt file written inside output_path.
        Each line contains --key value
    """
    txt_fpath = os.path.join(output_path, 'input_keyword_args.txt')
    with open(txt_fpath, 'w') as f:
        for key in sorted(arg_dict.keys()):
            f.write("--%s %s\n" % (key, str(arg_dict[key])))

def write_python_module_versions_to_txt(
        context_dict=None,
        output_path=None):
    """ Write .txt file to provided output_path dir with module info.

    Post condition
    --------------
    Writes plain text file called "modules_with_versions.txt" to disk.
    Each line contains name and version number of a python module.
    """
    watermark_string = util_watermark.make_string_of_reachable_modules_with_versions(
        context_dict=context_dict)
    txt_fpath = os.path.join(output_path, 'modules_with_versions.txt')
    with open(txt_fpath, 'w') as f:
        f.write(watermark_string)
