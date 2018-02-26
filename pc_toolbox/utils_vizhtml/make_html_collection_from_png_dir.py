import sys
import os
import argparse
import pandas as pd
import numpy as np
import glob
import shutil

from distutils.dir_util import mkpath
from collections import OrderedDict

TABLE_ROW_TEMPLATE_STR = (
    "<tr>" + 
    '<td class="col-md-2"><a href="$BASENAME_1">$DISPNAME_1</a></td>' + 
    '<td class="col-md-2"><a href="$BASENAME_2">$DISPNAME_2</a></td>' +
    '<td class="col-md-2"><a href="$BASENAME_3">$DISPNAME_3</a></td>' +
    "</tr>"
    )


def make_link_table_of_neighbor_basenames(
        field_uvals_dict, row_df, field):
    dispnames, basenames = make_neighbor_dispnames_and_basenames(
        field_uvals_dict, row_df, field)
    rem = len(basenames) % 3
    if rem == 1:
        dispnames = dispnames + ['', '']
        basenames = basenames + ['#', '#']
    elif rem == 2:
        dispnames = dispnames + ['']
        basenames = basenames + ['#']
    disp2base = dict(zip(dispnames, basenames))
    html_str = (
          '''<h3 align='center'> %s = %s </h3>\n''' % (field, row_df[field])
        + '''<table class="table" width="90%" border="1">\n'''
        )
    row_template_str = TABLE_ROW_TEMPLATE_STR
    for name_1, name_2, name_3 in zip(*[iter(dispnames)]*3):
        cur_row_str = row_template_str
        cur_row_str = cur_row_str\
            .replace("$DISPNAME_1", name_1)\
            .replace("$BASENAME_1", disp2base[name_1])
        cur_row_str = cur_row_str\
            .replace("$DISPNAME_2", name_2)\
            .replace("$BASENAME_2", disp2base[name_2])
        cur_row_str = cur_row_str\
            .replace("$DISPNAME_3", name_3)\
            .replace("$BASENAME_3", disp2base[name_3])
        html_str += cur_row_str + "\n"
    html_str += "</table>\n"
    return html_str


def make_basename(field_uvals_dict, row_df):
    basename_elts = list()
    for ii, field_name in enumerate(field_uvals_dict.keys()):
        basename_elts.append("%s=%s" % (field_name, row_df[field_name]))
    return "-".join(basename_elts) + ".html"

def make_neighbor_dispnames_and_basenames(
        field_uvals_dict, row_df, field_name):
    dispnames = field_uvals_dict[field_name]
    basenames = list()
    for uval in dispnames:
        neigh_df = row_df.copy()
        neigh_df[field_name] = uval
        basename = make_basename(field_uvals_dict, neigh_df)
        basenames.append(basename)
    return dispnames, basenames

SAFE_PRE_TAG = '<pre style="background-color:transparent;">'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--png_path')
    parser.add_argument('--html_output_path')
    parser.add_argument('--field_order')
    parser.add_argument('--within_field_order', type=str, default=None)
    parser.add_argument('--ncols', type=int, default=5)
    parser.add_argument('--n_rows_stop', type=int, default=None)
    parser.add_argument('--pprint_and_exit', type=int, default=0)
    args = parser.parse_args()
    for key in sorted(args.__dict__):
        print "--%s %s" % (key, args.__dict__[key])
    locals().update(args.__dict__)
    mkpath(html_output_path)

    # Read .csv
    # has fields
    png_fpath_list = glob.glob(os.path.join(png_path, '*.png'))

    row_dict_list = list()
    for fpath in png_fpath_list:
        row_dict = dict()
        head, tail = os.path.split(fpath)
        basename, ext = os.path.splitext(tail)
        basename_fields = basename.split("-")
        for field in basename_fields:
            if field.count("=") == 0:
                continue
            key, val = field.split("=")
            row_dict[key.upper()] = val
        row_dict['IMG_ABS_PATH'] = fpath
        row_dict['IMG_REL_PATH'] = os.path.basename(fpath)
        row_dict_list.append(row_dict)
    png_fpath_df = pd.DataFrame(row_dict_list)
       
    field_order = unicode(field_order).split(",")
    within_field_order_dict = dict()
    for key_and_order in unicode(within_field_order).split("/"):
        if key_and_order.count(":") != 1:
            continue
        parts = key_and_order.split(":")
        key = parts[0]
        order = parts[1]
        assert len(order) > 0
        assert len(key) > 0
        within_field_order_dict[key] = order.split(",")

    field_uvals_dict = OrderedDict()
    for field in field_order:
        uvals = np.unique(
            map(unicode,
                np.unique(png_fpath_df[field].values).tolist()),
            ).tolist()
        if field in within_field_order_dict:
            ord_uvals = within_field_order_dict[field]
            # Verify all ord_uvals appear in uvals
            assert np.setdiff1d(ord_uvals, uvals).size == 0
            field_uvals_dict[field] = ord_uvals
        elif len(uvals) > 1:
            # only keep fields with more than one unique value
            field_uvals_dict[field] = uvals
    field_order = field_uvals_dict.keys()

    n_rows_total = png_fpath_df.shape[0]
    n_rows_done = 0
    n_rows_skipped = 0
    html_paths = list()
    for _, row_df in png_fpath_df.iterrows():
        # Determine cur_html_path
        cur_html_basename = make_basename(
            field_uvals_dict,
            row_df)

        html_lines = []
        for field in field_order:
            cur_table_html = make_link_table_of_neighbor_basenames(
                field_uvals_dict, row_df, field)
            html_lines.append(cur_table_html)

        srcfile = row_df['IMG_REL_PATH']
        html_lines.append(
            SAFE_PRE_TAG + "\nSRCFILE:\n" + srcfile + "</pre><nbsp;>")
        if srcfile.endswith(".png"):
            img_tag_pat = '<img src="%s" class="img-fluid center-block" width="100%%" alt="Responsive image">'
            html_lines.append(img_tag_pat % srcfile)


        # Read in template and replace content
        with open("template.html", 'r') as f:
            all_html = ''.join(f.readlines())
        all_html = all_html.replace("$PAGE_TITLE", cur_html_basename)
        all_html = all_html.replace(
            "$PAGE_CONTENT",
            '\n'.join(html_lines))

        # Write current page content
        html_fpath = os.path.join(html_output_path, cur_html_basename)
        html_paths.append(html_fpath)
        with open(html_fpath, 'w') as f:
            f.write(all_html)
        # Copy over the img to the html_output_path
        shutil.copy(row_df['IMG_ABS_PATH'], html_output_path)

        n_rows_done += 1
        print "Wrote HTML %d/%d\n%s" % (n_rows_done, n_rows_total, html_fpath)

        if n_rows_stop and n_rows_done > n_rows_stop:
            break


    """
    # Now fix broken links
    topic_names = ['Gibbs_LDA', 'ourBP_sLDA', 'BP_sLDA', 'PC_sLDA', 'MED_sLDA']
    super_names = ['logistic_regr']
    for hpath in html_paths:
        bigstr = None
        with open(hpath, 'r') as f:
            bigstr = ''.join(f.readlines())

        # Sniff out the type of method
        leg_start = bigstr.find("LEGEND_NAME =")
        leg_end = bigstr.find("<", leg_start+1)
        cur_leg_str = bigstr[leg_start:leg_end].strip().split("=")[1].strip()
        assert cur_leg_str in topic_names or cur_leg_str in super_names

        start_link_str = '<a href="'
        startloc = bigstr.find(start_link_str)
        while startloc >= 0:
            endloc = bigstr.find('">', startloc)
            html_basename = bigstr[startloc+len(start_link_str):endloc]
            html_fullpath = os.path.join(html_output_path, html_basename)
            new_basename = html_basename
            if not os.path.exists(html_fullpath) and not html_basename == '#':
                if html_basename.count(cur_leg_str) > 0:
                    # Common case, within method links
                    new_basename = '#'
                elif html_basename.count("N_STATES=nan"):
                    # this is logistic making link to topic
                    arr_var = np.asarray(
                        field_uvals_dict['N_STATES'], dtype=np.float32)
                    other_val = np.nanmin(arr_var)
                    new_basename = html_basename.replace(
                        "N_STATES=nan",
                        "N_STATES=%s" % (other_val))
                elif html_basename.count('logistic_regr'):
                    sloc = html_basename.find('N_STATES=')
                    eloc = html_basename.find("_", sloc+4) # omit _ in N_STATES
                    if eloc < 0:
                        eloc = html_basename.find(".html")
                    key_and_val = html_basename[sloc:eloc]
                    new_basename = html_basename.replace(
                        key_and_val,
                        'N_STATES=nan')
                    for fval in ['0.05', '0.1', '0.2']:
                        new_basename = new_basename.replace(
                            "FRAC_LABELS=" + fval, "FRAC_LABELS=1.0")
                else:
                    new_basename = '#'

                if new_basename != '#':
                    new_fullpath = os.path.join(html_output_path, new_basename)
                    if os.path.exists(new_fullpath):
                        bigstr = bigstr.replace(
                            html_basename,
                            new_basename)
                    else:
                        bigstr = bigstr.replace(
                            html_basename,
                            '#')
                else:
                    bigstr = bigstr.replace(
                        html_basename,
                        '#')
                #if html_basename != new_basename:
                #    print '>BEF>', html_basename
                #    print '>AFT>', new_basename
            startloc = bigstr.find(start_link_str, startloc+1)

        with open(hpath, 'w') as f:
            f.write(bigstr)
    """