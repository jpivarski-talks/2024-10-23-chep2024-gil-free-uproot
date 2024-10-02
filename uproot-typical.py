import sys

sys.path.insert(0, "/home/jpivarski/storage/data/python-3.13-uproot")

import time
import threading

import uproot
import numpy as np


def in_thread(which, task_descriptions):
    if len(task_descriptions) == 1:
        tree1, tree2 = task_descriptions[which]

        array = tree1.arrays()
        del array

        array = tree2.arrays()
        del array

    else:
        tree, entry_start, entry_stop = task_descriptions[which]

        array = tree.arrays(entry_start=entry_start, entry_stop=entry_stop)
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


if NUM_THREADS == 1:
    task_descriptions = [(tree1, tree2)]

else:
    step_size = (tree1.num_entries + tree2.num_entries) // NUM_THREADS

    tree1_task_descriptions = []
    for entry_start in range(0, tree1.num_entries, step_size):
        entry_stop = min(entry_start + step_size, tree1.num_entries)
        tree1_task_descriptions.append((tree1, entry_start, entry_stop))

    tree2_task_descriptions = []
    for entry_start in range(0, tree2.num_entries, step_size):
        entry_stop = min(entry_start + step_size, tree2.num_entries)
        tree2_task_descriptions.append((tree2, entry_start, entry_stop))

    def merge_last(task_descriptions):
        if len(task_descriptions) > 1:
            tree, l1_start, l1_stop = task_descriptions[-1]
            tree, l2_start, l2_stop = task_descriptions[-2]
            del task_descriptions[-1]
            task_descriptions[-1] = (tree, l2_start, l1_stop)

    if len(tree1_task_descriptions) + len(tree2_task_descriptions) > NUM_THREADS:
        if (
            tree1_task_descriptions[-1][2] - tree1_task_descriptions[-1][1]
            < tree2_task_descriptions[-1][2] - tree2_task_descriptions[-1][1]
        ):
            merge_last(tree1_task_descriptions)
            if len(tree1_task_descriptions) + len(tree2_task_descriptions) > NUM_THREADS:
                merge_last(tree2_task_descriptions)

        else:
            merge_last(tree2_task_descriptions)
            if len(tree1_task_descriptions) + len(tree2_task_descriptions) > NUM_THREADS:
                merge_last(tree1_task_descriptions)

    assert len(tree1_task_descriptions) + len(tree2_task_descriptions) == NUM_THREADS

    for tree, _, _ in tree1_task_descriptions:
        assert tree is tree1
    for tree, _, _ in tree2_task_descriptions:
        assert tree is tree2

    for i in range(len(tree1_task_descriptions) - 1):
        assert tree1_task_descriptions[i][2] == tree1_task_descriptions[i + 1][1]
    for i in range(len(tree2_task_descriptions) - 1):
        assert tree2_task_descriptions[i][2] == tree2_task_descriptions[i + 1][1]

    assert tree1_task_descriptions[0][1] == 0
    assert tree1_task_descriptions[-1][2] == tree1.num_entries
    assert tree2_task_descriptions[0][1] == 0
    assert tree2_task_descriptions[-1][2] == tree2.num_entries

    task_descriptions = tree1_task_descriptions + tree2_task_descriptions

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
