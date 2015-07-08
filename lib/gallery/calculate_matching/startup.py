#FV-Benchmark

import sys
import os

sys.path[0] = '/'.join(os.getcwd().split('/')[:-2]) + "/lib"
print sys.path[0]
os.environ['PYTHONPATH'] = ':'.join(sys.path)
