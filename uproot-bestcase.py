import sys

sys.path.insert(0, "/home/jpivarski/storage/data/python-3.13-uproot")

import time
import threading
import queue

import uproot
import numpy as np


def in_thread(which, task_descriptions):
    while True:
        task_description = task_descriptions.get()

        if task_description is None:
            break

        branch, entry_start, entry_stop = task_description

        array = branch.array(entry_start=entry_start, entry_stop=entry_stop)
        del array


NUM_THREADS = int(sys.argv[1])
SOURCE = uproot.MemmapSource if sys.argv[2] == "mm" else None

# https://opendata.cern.ch/record/12365
tree1 = uproot.open(
    "/home/jpivarski/Downloads/Run2012B_DoubleMuParked.root:Events",
    array_cache=None,
    handler=SOURCE,
)

# https://opendata.cern.ch/record/12366
tree2 = uproot.open(
    "/home/jpivarski/Downloads/Run2012C_DoubleMuParked.root:Events",
    array_cache=None,
    handler=SOURCE,
)

task_descriptions = queue.Queue()
for tree in [tree1, tree2]:
    for branch in tree.branches:
        for basketid in range(branch.num_baskets):
            task_descriptions.put((branch,) + branch.basket_entry_start_stop(basketid))

for _ in range(100):
    task_descriptions.put(None)

threads = []
for which in range(NUM_THREADS):
    threads.append(threading.Thread(target=in_thread, args=(which, task_descriptions)))

start_timer = time.perf_counter()

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

stop_timer = time.perf_counter()

print(NUM_THREADS, type(tree1.file.source).__name__, stop_timer - start_timer)

