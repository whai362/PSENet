import subprocess
import os
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if subprocess.call(['make', '-C', BASE_DIR]) != 0:
    raise RuntimeError('Cannot compile pse: {}'.format(BASE_DIR))

from .adaptor import pse as cpse
def pse(polys, min_area):
    # start = time.time()
    ret = np.array(cpse(polys, min_area), dtype='int32')
    # end = time.time()
    # print (end - start), 's'
    return ret

