import argparse
import os
import numpy as np
import scipy.sparse
from distutils.dir_util import mkpath
from sklearn.externals import joblib

from sscape.utils_io import load_csr_matrix, save_csr_matrix
import bow_dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", default=".", type=str)
    parser.add_argument(
        "--output_path", default="./frac_labels=$frac_labels_train/", type=str)
    parser.add_argument("--frac_labels_train", default=0.2, type=float)
    args = parser.parse_args()
    locals().update(vars(args))

    dataset_path = os.path.abspath(dataset_path)
    output_path = os.path.abspath(output_path.replace(
        '$frac_labels_train', '%.3f' % frac_labels_train))
    for key in sorted(vars(args).keys()):
        print '--%s %s' % (key, locals()[key])

    for split_name in ['train', 'valid', 'test']:
        dataset_info = bow_dataset.load_dataset(
            dataset_path, split_name,
            frac_labels_train=frac_labels_train)
        dataset = dataset_info['dataset']

        if 'y_rowmask' in dataset:
            dataset['y_DC'][dataset['y_rowmask']==0] = np.nan

        print bow_dataset.describe_bow_dataset(
            dataset=dataset,
            dataset_name="haystack: %s set" % split_name,
            label_list=dataset_info.get('label_list', None))
        bow_dataset.save_dataset(
            dataset=dataset,
            output_path=output_path,
            split_name=split_name)

    print dataset_info.keys()
    with open(os.path.join(output_path, 'X_colnames.txt'), 'w') as f:
        for xname in dataset_info['vocab_list']:
            f.write("%s\n" % xname)
    with open(os.path.join(output_path, 'Y_colnames.txt'), 'w') as f:
        for name in dataset_info['label_list']:
            f.write("%s\n" % name)

        

