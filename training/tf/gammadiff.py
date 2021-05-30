#!/home/vandertic/tf-gpu-1.13.1/bin/python

import sys
import os
import numpy as np

for arg in sys.argv[1:]:
    base = os.path.basename(arg)
    name, ext = os.path.splitext(base)
    if ext != '.gamma':
#        print(f'No: {ext}')
        continue
    print(name)

    file = open(arg, 'r')

    i = 0

    line = file.readline()

    while line != "":
        i += 1
        line = line.split()
        line = [float(x) for x in line]
        line = np.array(line)
        norm = np.linalg.norm(np.log(line))
        diff = line - 1
        print(i, len(diff), diff.min(), diff.max(), norm)
    
        line = file.readline()

