import sys
import os
import argparse
import pandas as pd
import numpy as np
import glob

from distutils.dir_util import mkpath
from collections import OrderedDict

from utils_viz_topic_model import show_topics_and_weights

TABLE_ROW_TEMPLATE_STR = (
    "<tr>" + 
    '<td class="col-md-2"><a href="$BASENAME_1">$DISPNAME_1</a></td>' + 
    '<td class="col-md-2"><a href="$BASENAME_2">$DISPNAME_2</a></td>' +
    '<td class="col-md-2"><a href="$BASENAME_3">$DISPNAME_3</a></td>' +
    "</tr>"
    )

topic_names = ['Gibbs_LDA', 'ourBP_sLDA', 'BP_sLDA', 'PC_sLDA', 'MED_sLDA']
super_names = [
    'logistic_regr',
    'linear_regr',
    'linear_regression',
    'extra_trees',
    'rand_forest']


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
            .replace("$DISPNAME_1", str(name_1))\
            .replace("$BASENAME_1", disp2base[name_1])
        cur_row_str = cur_row_str\
            .replace("$DISPNAME_2", str(name_2))\
            .replace("$BASENAME_2", disp2base[name_2])
        cur_row_str = cur_row_str\
            .replace("$DISPNAME_3", str(name_3))\
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
    parser.add_argument('--snapshot_csv_path')
    parser.add_argument('--field_order')
    parser.add_argument('--html_output_path')
    parser.add_argument('--ncols', type=int, default=5)
    parser.add_argument('--n_words_per_topic', type=int, default=15)
    parser.add_argument('--n_chars_per_word', type=int, default=16)
    parser.add_argument('--n_rows_stop', type=int, default=None)
    parser.add_argument('--show_enriched_words', type=int, default=0)
    parser.add_argument('--show_longer_words_via_tooltip', type=int, default=0)
    parser.add_argument('--pprint_snapshot_list_and_exit', type=int, default=0)
    parser.add_argument('--add_bias_term_to_w_CK', type=float, default=0.0)
    args = parser.parse_args()
    for key in sorted(args.__dict__):
        print "--%s %s" % (key, args.__dict__[key])
    locals().update(args.__dict__)
    mkpath(html_output_path)

    # Read .csv
    # has fields
    csv_df = pd.read_csv(snapshot_csv_path)

    field_order = field_order.split(",")
    field_uvals_dict = OrderedDict()
    for field in field_order:
        assert field in csv_df
        # Try to convert to numeric type
        # but if it fails, keep as a string
        values = pd.to_numeric(csv_df[field], errors='ignore')
        try:
            uvals = np.unique(
                map(np.float64,
                    np.unique(values).tolist()),
                )
            finite_bmask = np.isfinite(uvals)
            uvals = pd.to_numeric(uvals, downcast='signed')
            csv_df[field] = csv_df[field].astype(uvals.dtype)
            if not np.all(finite_bmask):
                uvals = uvals[finite_bmask].tolist() + ['nan']
            else:
                uvals = uvals.tolist()
        except Exception:
            uvals = np.unique(
                map(unicode,
                    np.unique(values).tolist()),
                ).tolist()
            csv_df[field] = csv_df[field].astype(unicode)
        field_uvals_dict[field] = uvals
        print("%s: %s" % (field, ','.join(['%s' % a for a in uvals])))
        # TODO if any field has only one val across all rows, just remove it
    field_order = field_uvals_dict.keys()

    # Parse each row of the csv file
    n_rows_total = csv_df.shape[0]
    n_rows_done = 0
    n_rows_skipped = 0
    html_paths = list()
    for _, row_df in csv_df.iterrows():
        # Determine cur_html_path
        cur_html_basename = make_basename(
            field_uvals_dict,
            row_df)

        html_lines = []
        for field in field_order:
            cur_table_html = make_link_table_of_neighbor_basenames(
                field_uvals_dict, row_df, field)
            html_lines.append(cur_table_html)

        srcfile = row_df['SNAPSHOT_SRCFILE']

        html_lines.append(
            SAFE_PRE_TAG + "\nSNAPSHOT_SRCFILE:\n" + srcfile + "</pre><nbsp;>")

        if srcfile.endswith(".txt"):
            html_lines.append(SAFE_PRE_TAG)
            with open(srcfile, 'r') as f:
                html_lines.append(''.join(f.readlines()))
            html_lines.append("</pre>")
        else:
            label_name = row_df["LABEL_NAME"]
            snapshot_path = row_df['SNAPSHOT_SRCFILE']
            txtsrc_path = row_df['TXTSRCFILES_PATH']

            assert os.path.exists(snapshot_path)
            assert os.path.exists(txtsrc_path)

            vocab_list = np.atleast_1d(np.loadtxt(
                os.path.join(txtsrc_path, 'X_colnames.txt'),
                dtype=unicode)).tolist()
            label_list = np.atleast_1d(np.loadtxt(
                os.path.join(txtsrc_path, 'Y_colnames.txt'),
                dtype=unicode)).tolist()
            html_str = show_topics_and_weights(
                snapshot_path=snapshot_path,
                sort_by='w_CK',
                add_bias_term_to_w_CK=add_bias_term_to_w_CK,
                vocab_list=vocab_list,
                y_ind=label_list.index(label_name),
                do_html=True,
                show_enriched_words=show_enriched_words,
                n_top_words=n_words_per_topic,
                wordSizeLimit=n_chars_per_word,
                show_longer_words_via_tooltip=show_longer_words_via_tooltip,
                proba_fmt_str='%.3f',
                ncols=ncols)
            html_lines.append(html_str)

        with open("template.html", 'r') as f:
            all_html = ''.join(f.readlines())
        all_html = all_html.replace("$PAGE_TITLE", cur_html_basename)
        all_html = all_html.replace(
            "$PAGE_CONTENT",
            '\n'.join(html_lines))

        html_fpath = os.path.join(html_output_path, cur_html_basename)
        html_paths.append(html_fpath)
        with open(html_fpath, 'w') as f:
            f.write(all_html)

        n_rows_done += 1
        print "Wrote HTML %d/%d\n%s" % (n_rows_done, n_rows_total, html_fpath)

        if n_rows_stop and n_rows_done > n_rows_stop:
            break
    if n_rows_skipped > 0 and pprint_snapshot_list_and_exit:
        sys.exit(0)

    # Now fix broken links
    for hpath in html_paths:
        bigstr = None
        with open(hpath, 'r') as f:
            bigstr = ''.join(f.readlines())

        # Sniff out the type of method
        leg_start = bigstr.find("LEGEND_NAME =")
        leg_end = bigstr.find("<", leg_start+1)
        cur_leg_str = bigstr[leg_start:leg_end].strip().split("=")[1].strip()
        assert cur_leg_str in topic_names or cur_leg_str in super_names

        # Loop thru each link in the current html file
        start_link_str = '<a href="'
        startloc = bigstr.find(start_link_str)
        while startloc >= 0:
            endloc = bigstr.find('">', startloc)
            html_basename = bigstr[startloc+len(start_link_str):endloc]
            html_fullpath = os.path.join(html_output_path, html_basename)
            new_basename = html_basename
            if not os.path.exists(html_fullpath) and not html_basename == '#':

                is_super = sum([html_basename.count(s) for s in super_names])
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
                elif is_super:
                    sloc = html_basename.find('N_STATES=')
                    eloc = html_basename.find("_", sloc+4) # omit _ in N_STATES
                    if eloc < 0:
                        eloc = html_basename.find(".html")
                    key_and_val = html_basename[sloc:eloc]
                    new_basename = html_basename.replace(
                        key_and_val,
                        'N_STATES=nan')
                    """
                    for fval in ['0.05', '0.1', '0.2']:
                        new_basename = new_basename.replace(
                            "FRAC_LABELS=" + fval, "FRAC_LABELS=1.0")
                    """
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
    print "DONE. Find created HTML files in dir:"
    print html_output_path