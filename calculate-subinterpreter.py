import sys

sys.path.insert(0, "/home/jpivarski/storage/data/python-3.13-uproot")

import time
import array
import ctypes
import test.support.interpreters.queues

import numpy as np
import awkward as ak
import uproot


def in_subinterp():
    # Need to re-import; this is in its own little world...

    import sys
    print(f"START {which}", file=sys.stderr)

    from math import sqrt, cosh, cos
    import ctypes
    import test.support.interpreters.queues

    # Wouldn't it be nice to use NumPy?
    offsets = (ctypes.c_int64 * (N + 1)).from_address(ptr_offsets)
    pt = (ctypes.c_float * offsets[-1]).from_address(ptr_pt)
    eta = (ctypes.c_float * offsets[-1]).from_address(ptr_eta)
    phi = (ctypes.c_float * offsets[-1]).from_address(ptr_phi)
    mass = (ctypes.c_float * N).from_address(ptr_mass)

    # Do something computationally expensive. How about dimuon mass?
    for event in range(start, stop):
        max_mass = 0
        for i in range(offsets[event], offsets[event + 1]):
            pt1 = pt[i]
            eta1 = eta[i]
            phi1 = phi[i]
            for j in range(i + 1, offsets[event + 1]):
                pt2 = pt[j]
                eta2 = eta[j]
                phi2 = phi[j]
                m = sqrt(2*pt1*pt2*(cosh(eta1 - eta2) - cos(phi1 - phi2)))
                if m > max_mass:
                    max_mass = m
        mass[event] = max_mass

    # Inform the outer world that the work is done.
    notify_when_done = test.support.interpreters.queues.Queue(queue_id)
    notify_when_done.put(which)


# https://opendata.cern.ch/record/12365
with uproot.open("~/Downloads/Run2012B_DoubleMuParked.root:Events") as tree:
    arrays1 = tree.arrays(["Muon_pt", "Muon_eta", "Muon_phi"])

# https://opendata.cern.ch/record/12366
with uproot.open("~/Downloads/Run2012C_DoubleMuParked.root:Events") as tree:
    arrays2 = tree.arrays(["Muon_pt", "Muon_eta", "Muon_phi"])

arrays = ak.concatenate([arrays1, arrays2])
del arrays1, arrays2

N = len(arrays)

offsets = arrays["Muon_pt"].layout.offsets.data
pt = arrays["Muon_pt"].layout.content.data
eta = arrays["Muon_eta"].layout.content.data
phi = arrays["Muon_phi"].layout.content.data

mass = np.zeros(N, np.float32)

NUM_THREADS = int(sys.argv[1])
start_stop = np.linspace(0, N, NUM_THREADS + 1).astype(int).tolist()

notify_when_done = test.support.interpreters.queues.create()

time.sleep(1)

print("START", file=sys.stderr)

start_timer = time.perf_counter()

subinterps = []
for which in range(NUM_THREADS):
    start, stop = start_stop[which], start_stop[which + 1]

    subinterp = test.support.interpreters.create()
    subinterp.prepare_main({
        "which": which,
        "queue_id": notify_when_done.id,
        "N": N,
        "start": start,
        "stop": stop,
        "ptr_offsets": offsets.ctypes.data,
        "ptr_pt": pt.ctypes.data,
        "ptr_eta": eta.ctypes.data,
        "ptr_phi": phi.ctypes.data,
        "ptr_mass": mass.ctypes.data,
    })
    subinterp.call_in_thread(in_subinterp)
    subinterps.append(subinterp)

done = []
while len(done) != len(subinterps):
    done.append(notify_when_done.get())

for subinterp in subinterps:
    subinterp.close()

stop_timer = time.perf_counter()

print("STOP", file=sys.stderr)

print(NUM_THREADS, stop_timer - start_timer)

time.sleep(1)
