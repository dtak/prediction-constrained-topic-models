import os

import utils_io
import utils_data
import utils_snapshots

import model_slda

# TODO discard this line
# calc_nef_map_pi_DK = model_slda.calc_nef_map_pi_DK

PC_REPO_DIR = os.path.sep.join(
    os.path.abspath(__file__).split(os.path.sep)[:-2])

## Create version attrib
__version__ = None
version_txt_path = os.path.join(PC_REPO_DIR, 'version.txt')
if os.path.exists(version_txt_path):
   with open(version_txt_path, 'r') as f:
        __version__ = f.readline().strip()

## Create requirements attrib
__requirements__ = None
reqs_txt_path = os.path.join(PC_REPO_DIR, 'requirements.txt')
if os.path.exists(reqs_txt_path):
   with open(reqs_txt_path, 'r') as f:
        __requirements__ = []
        for line in f.readlines():
            __requirements__.append(line.strip())




