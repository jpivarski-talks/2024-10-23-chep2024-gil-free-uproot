import sys

sys.path.insert(0, "/home/jpivarski/storage/data/python-3.13-uproot")

import time
from concurrent.futures import ThreadPoolExecutor

import uproot
import numpy as np


NUM_THREADS = int(sys.argv[1])
SOURCE = uproot.MemmapSource if sys.argv[2] == "mm" else None

executor = ThreadPoolExecutor(max_workers=NUM_THREADS)

start_timer = time.perf_counter()

for array in uproot.iterate(
    [
        "/home/jpivarski/Downloads/Run2012B_DoubleMuParked.root:Events",
        "/home/jpivarski/Downloads/Run2012C_DoubleMuParked.root:Events",
    ],
    decompression_executor=executor,
    interpretation_executor=executor,
    handler=SOURCE,
    step_size="1 GB",
):
    del array

stop_timer = time.perf_counter()

print(NUM_THREADS, SOURCE, stop_timer - start_timer)
