'''
utils_viz_top_words.py

'''
import numpy as np
import argparse
import os
import sys

STYLE_HEADER_HTML_STR = """
<style>
pre.num {line-height:13px; font-size:10px; display:inline; color:lightgray; background-color:transparent; border:0px}
pre.word {line-height:13px; font-size:13px; display:inline; color:black; background-color:transparent; border:0px}
h2.uid {line-height:20px; font-size:16px;
    text-align:left; padding:0px; margin:0px;
    color:#e2dcdc; display: inline;}
h2.label {line-height:20px; font-size:20px;
    text-align:left; padding:0px; margin:0px;
    color:gray; display:inline;}
td {padding-top:5px; padding-bottom:5px;}
table { page-break-inside:auto }
tr { page-break-inside:avoid; page-break-after:auto }
</style>
"""


def make_top_words_html_from_topics(
        topics_KV,
        vocab_list=None,
        order=None,
        uids_K=None,
        ncols=5,
        n_words_per_topic=10,
        max_topics_to_display=100,
        proba_fmt_str='%.4f',
        n_chars_per_word=30,
        show_longer_words_via_tooltip=0,
        xlabels=None,
        **kwargs):
    K, V = topics_KV.shape
    if order is None:
        order = np.arange(K)
    htmllines = list()
    htmllines.append(STYLE_HEADER_HTML_STR)
    htmllines.append('<table>')
    for posID, k in enumerate(order[:max_topics_to_display]):
        if posID % ncols == 0:
            htmllines.append('  <tr>')

        if uids_K is None:
            uid = k + 1
        else:
            uid = uids_K[k]
        #k = k[0]
        if xlabels is None:
            titleline = '<h2>%4d/%d</h2>' % (
                uid, K)
        else:
            titleline = (
                '<h2 class="uid">%4d/%d</h2>' +
                '<h2 class="label">%10s</h2><br />') % (
                uid, K, xlabels[k])
        htmllines.append('    <td>' + titleline)
        htmllines.append('    ')

        # want to use fmtr like "%-20s" to force 20 chars of whitespace
        fixed_width_str__fmtr = "%" + "-" + str(n_chars_per_word) + "s"
        htmlPattern = \
            '<pre class="num">' + proba_fmt_str + ' ' + \
            '</pre><pre class="word">' \
            + fixed_width_str__fmtr + "</pre>"
        topIDs = np.argsort(-1 * topics_KV[k])[:n_words_per_topic]
        for topID in topIDs:
            dataline = htmlPattern % (
                topics_KV[k, topID],
                vocab_list[topID][:n_chars_per_word])
            if show_longer_words_via_tooltip:
                if len(vocab_list[topID]) > n_chars_per_word:
                    dataline = dataline.replace(
                        '<pre class="word">',
                        '<pre class="word" title="%s">' % vocab_list[topID],                            
                        )
            htmllines.append(dataline + "<br />")
        htmllines.append('    </td>')

        if posID % ncols == ncols - 1:
            htmllines.append(' </tr>')
    htmllines.append('</table>')
    htmlstr = '\n'.join(htmllines)
    return htmlstr


