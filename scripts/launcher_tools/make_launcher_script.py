#!/usr/bin/env python
"""
make_launcher_script.py

User-Executable script that creates temp launcher script file

Usage
-----
$ python make_launcher_script.py
<prints name of launcher file to stdout>
"""
import os
import distutils.spawn
import tempfile

DEFAULT_KEYS = [
    'XHOST_JOB_NAME',
    'XHOST_MACHINE_NAME',
    'XHOST_LOG_DIR',
    'XHOST_FIRSTTASK',
    'XHOST_NTASKS',
    'XHOST_MEM_MB',
    'XHOST_SWP_MB',
    'XHOST_TIME_HR',
    ]

def set_default_environment():
    if 'XHOST_BASH_EXE' not in os.environ:
        raise ValueError("Need to define env var: XHOST_BASH_EXE")
    assert os.path.exists(os.environ['XHOST_BASH_EXE'])

    if 'XHOST_LOG_DIR' not in os.environ:
        raise ValueError("Need to define env var: XHOST_LOG_DIR")
    assert os.path.exists(os.environ['XHOST_LOG_DIR'])

    if 'XHOST_NTASKS' not in os.environ:
        os.environ['XHOST_NTASKS'] = '1'
    if 'XHOST_FIRSTTASK' not in os.environ:
        os.environ['XHOST_FIRSTTASK'] = '1'
    if 'XHOST_MEM_MB' not in os.environ:
        os.environ['XHOST_MEM_MB'] = '5000'
    if 'XHOST_SWP_MB' not in os.environ:
        os.environ['XHOST_SWP_MB'] = '5000'
    if 'XHOST_MACHINE_NAME' not in os.environ:
        os.environ['XHOST_MACHINE_NAME'] = 'liv'
    if 'XHOST_JOB_NAME' not in os.environ:
        os.environ['XHOST_JOB_NAME'] = 'my_job'
    if 'XHOST_TIME_HR' not in os.environ:
        os.environ['XHOST_TIME_HR'] = '24'

def detect_template_ext_for_current_system():
    if distutils.spawn.find_executable("sacct"):
        return "slurm"
    elif distutils.spawn.find_executable("bjobs"):
        return "lsf"
    elif distutils.spawn.find_executable("qstat"):
        return "sge"
    raise ValueError("Unknown grid system")

def make_launcher_script_file():
    """ Create temporary file for launching job on grid system

    Post Condition
    --------------
    Temporary file written to /tmp/ or similar via tempfile module

    Returns
    -------
    fpath : string
        Valid path to temporary file
    """

    ext_str = detect_template_ext_for_current_system()
    template_fpath = os.path.join(
        os.path.expandvars("$PC_REPO_DIR/scripts/launcher_tools/"),
        "template.%s" % ext_str)
    with open(template_fpath, "r") as f:
        template_lines = f.readlines()

    launcher_f = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="launcher_for_%s_" % os.environ['USER'],
        suffix="." + ext_str,
        delete=False)
    for line in template_lines:
        for key in DEFAULT_KEYS:
            line = line.replace("$" + key, os.environ[key])
        line = line.replace(
            '$XHOST_BASH_EXE',
            os.path.abspath(os.environ['XHOST_BASH_EXE']))
        launcher_f.write(line)
    launcher_f.close()
    return os.path.abspath(launcher_f.name)


if __name__ == "__main__":
    set_default_environment()
    print(make_launcher_script_file())
