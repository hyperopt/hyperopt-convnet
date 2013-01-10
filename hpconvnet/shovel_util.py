import logging
import os
import subprocess
import sys
import time

def configure_logging(level=logging.INFO, stream=sys.stdout, prefix=''):
    if isinstance(level, basestring):
        level = getattr(logging, level)
    root_handler = logging.StreamHandler(stream)
    root_formatter = logging.Formatter(
        prefix + '%(levelname)s (%(name)s): %(message)s'
        )
    root_handler.setFormatter(root_formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(root_handler)

    logging.getLogger('shovel').propagate = False
    logging.getLogger('skimage').propagate = False


def launch_workers_helper(host, port, dbname, N, walltime, rsync_data_local,
                          mem=None):
    text = """#!/bin/bash
    %(rsync_data_local)s
    . VENV/eccv12/bin/activate
    VENV/eccv12/src/eccv12/hyperopt/bin/hyperopt-mongo-worker \
        --mongo=%(host)s:%(port)s/%(dbname)s \
        --workdir=/scratch_local/eccv12.workdir \
        --reserve-timeout=180.0 \
        --max-consecutive-failures=4
    """ % locals()

    qsub_script_name = '.worker.sh.%.3f' % time.time()

    script = open(qsub_script_name, 'w')
    script.write(text)
    script.close()

    subprocess.check_call(['chmod', '+x', qsub_script_name])
    qsub_cmd = ['qsub', '-lnodes=1:gpus=1', '-lwalltime=%s' % walltime]
    if mem is not None:
        qsub_cmd.append('-lmem=%s' % mem)
    qsub_cmd.extend(
            ['-e', os.path.expanduser('~/.qsub/%s.err' % qsub_script_name)])
    qsub_cmd.extend(
            ['-o', os.path.expanduser('~/.qsub/%s.out' % qsub_script_name)])
    if int(N) > 1:
        qsub_cmd.extend(['-t', '1-%s' % N])
    qsub_cmd.append(qsub_script_name)
    print qsub_cmd
    subprocess.check_call(qsub_cmd)



