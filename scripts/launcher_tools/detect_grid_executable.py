from __future__ import print_function
import os

import make_launcher_script as mls

if __name__ == '__main__':
    ext_str = mls.detect_template_ext_for_current_system()

    if ext_str == 'sge':
        print('qsub')
    elif ext_str == 'lsf':
        print('bsub')
    elif ext_str == 'slurm':
        print('sbatch')
    else:
        raise ValueError("Unrecognized extension: %s" % ext_str)
