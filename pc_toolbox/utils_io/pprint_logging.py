import logging
import os
import sys

RootLog = None

def pprint(msg_str='', level=logging.INFO):
    global RootLog
    if RootLog is None:
        print msg_str
    else:
        RootLog.log(level, msg_str)

def config_pprint_logging(
        output_path='/tmp/',
        do_write_txtfile=True,
        do_write_stdout=True,
        txtfile='stdout.txt',
        ):
    global RootLog
    RootLog = logging.getLogger('pprint_logging')
    RootLog.handlers = []
    RootLog.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(message)s')
    # Config logger to save transcript of log messages to plain-text file
    if do_write_txtfile:
        fh = logging.FileHandler(os.path.join(output_path, txtfile))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        RootLog.addHandler(fh)
    # Config logger that can write to stdout
    if do_write_stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        RootLog.addHandler(ch)

    # Config null logger, avoids error messages about no handler existing
    if not do_write_txtfile and not do_write_stdout:
        RootLog.addHandler(logging.NullHandler())

    '''
    # Prepare special logs if we are running on the Brown CS grid
    try:
        jobID = int(os.getenv('JOB_ID'))
    except TypeError:
        jobID = 0
    if jobID > 0:
        Log.info('SGE Grid Job ID: %d' % (jobID))

        if 'SGE_STDOUT_PATH' in os.environ:
            # Create symlinks to captured stdout, stdout in output directory
            os.symlink(os.getenv('SGE_STDOUT_PATH'),
                       os.path.join(taskoutpath, 'stdout'))
            os.symlink(os.getenv('SGE_STDERR_PATH'),
                       os.path.join(taskoutpath, 'stderr'))

            with open(os.path.join(taskoutpath, 'GridInfo.txt'), 'w') as f:
                f.write(str(jobID) + "\n")
                f.write(str(taskid) + "\n")
                f.write('stdout: ' + os.getenv('SGE_STDOUT_PATH') + "\n")
                f.write('stderr: ' + os.getenv('SGE_STDERR_PATH') + "\n")
    return jobID
    '''
