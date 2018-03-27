import numpy as np

from pc_toolbox.utils_io import make_percentile_str

def make_readable_summary_for_pi_DK_estimation(
        n_docs=None,
        elapsed_time_sec=None,
        n_docs_completed=None,
        n_docs_converged=None,
        n_docs_restarted=None,
        iters_per_doc=None,
        dist_per_doc=None,
        loss_per_doc=None,
        step_size_per_doc=None,
        converged_per_doc=None,
        n_active_per_doc=None,
        restarts_per_doc=None,
        pi_converge_thr=None,
        **unused_kws):
    if n_docs_completed is None:
        n_docs_completed = n_docs
    msg = "completed %d/%d docs" % (n_docs_completed, n_docs)
    if elapsed_time_sec is not None:
        msg += " after %7.2f sec" % (elapsed_time_sec)
    if converged_per_doc is not None:
        n_docs_converged = np.sum(converged_per_doc[:n_docs_completed])
    if n_docs_converged is not None:
        msg += " %6d not converged" % (
            n_docs_completed - n_docs_converged)
    if pi_converge_thr is not None:
        msg += " %6.2g conv_thr" % pi_converge_thr
    if n_docs_restarted is not None:
        msg += " %6d restarted" % (n_docs_restarted)
    if iters_per_doc is not None:
        msg += "\n         iters / doc: %s" % (
            make_percentile_str(
                iters_per_doc[:n_docs_completed],
                fmt_str='%7d'))
    if dist_per_doc is not None:
        msg += "\n       l1 dist / doc: %s" % (
            make_percentile_str(
                dist_per_doc[:n_docs_completed],
                fmt_str='%7.2g'))
    if step_size_per_doc is not None:
        msg += "\n  pi_step_size / doc: %s" % (
            make_percentile_str(
                step_size_per_doc[:n_docs_completed],
                fmt_str='%7.2g'))
    if loss_per_doc is not None:
        msg += "\n          loss / doc: %s" % (
            make_percentile_str(
                loss_per_doc[:n_docs_completed],
                fmt_str='% 7.4g'))
    if restarts_per_doc is not None:
        msg += "\n      restarts / doc: %s" % (
            make_percentile_str(
                restarts_per_doc[:n_docs_completed],
                fmt_str='%7d'))
    if n_active_per_doc is not None:
        msg += "\n active topics / doc: %s" % (
            make_percentile_str(
                n_active_per_doc[:n_docs_completed],
                fmt_str='%7d'))
    return msg
