#!/home/vandertic/tf-gpu-1.13.1/bin/python

import sys
import os
from tfprocess import TFProcess

tfprocess = TFProcess(12, 256, 0.05, 1000, 1000, 1000, 5, 1, 1, 0, 0, 1, 2, 0)
tfprocess.init(128, logbase="../gamma", macrobatch=4)

for arg in sys.argv[1:]:
    base = os.path.basename(arg)
    name, ext = os.path.splitext(base)
    if ext != '.meta':
#        print(f'No: {ext}')
        continue
    print(name)
    rest = os.path.splitext(arg)[0]
    tfprocess.restore(rest)
    tfprocess.save_gamma_weights(rest + ".gamma")
