"""
This script runs the BVEX model in a standalone mode. Key parameters are specified
in namelist.py. One complete run is split into a number of relays, which are
comprised by sprints. One time slice is saved after one sprint (multiple time
steps), and one relay outputs one NetCDF file with multiple time slices.
"""

import time
import numpy.fft as nfft
from bvex import *


# timers for performance evaluation
timerSetup = 0.0
timerRelay1 = 0.0  # first relay
timerRelayO = 0.0  # other relays
timerWrite = 0.0
walltime = time.time()


######################
# Main body for the model integration and data saving

# Create array to save data (for ML training these need to be stacked/concatenated jnp arrays)
numRec = np.min([1000, nSprints]).astype(int)
zetaRec = np.empty((numRec, ny, nx), dtype=np.float32)
psiRec = np.empty((numRec, ny, nx), dtype=np.float32)
timeRec = np.empty((numRec,), dtype=np.float32)

# Start the integration
timerSetup = time.time() - walltime + timerSetup
walltime = time.time()

q0, _, Kx, Ky, Del2, _, _, x, y = setup_ic_grid(shearFrac, randSeed)

if isRestart: 
    dsFile = nc4.Dataset(restartFile)
    zeta = np.copy(dsFile.variables["zeta"]).astype("float32")
    q0 = np.transpose(zeta[-1]).astype("float32")
    ncTime = np.copy(dsFile.variables["time"]).astype("float32")
    qNow = q0 
    tNow = ncTime[-1]
else:
    qNow = q0 
    tNow = 0.0


it = 0
iTotal = np.around(t_max / dt).astype(int)

while it < iTotal:
    qNow, tNow, qHist = relay(qNow, tNow)

    if it < (nSteps * nSprints - 1):
        timerRelay1 = time.time() - walltime + timerRelay1
        walltime = time.time()
    else:
        timerRelayO = time.time() - walltime + timerRelayO
        walltime = time.time()

    it = it + nSteps * nSprints

    # record data
    qHatHist = nfft.fft2(qHist)
    pHatHist = qHatHist / Del2
    pHist = np.real(nfft.ifft2(pHatHist))
    u = np.real(nfft.ifft2(-1j * Ky * pHatHist))
    v = np.real(nfft.ifft2(+1j * Kx * pHatHist))
    cfl = np.sqrt(np.max(u ** 2 + v ** 2)) * dt / dx
    print(("Relay #%4i;  CFL =%5.2f " % (int(it / nSteps / nSprints), cfl)))
    zetaRec = np.transpose(qHist, axes=[0, 2, 1]).astype("float32")
    psiRec = np.transpose(pHist, axes=[0, 2, 1]).astype("float32")
    timeRec = np.around(
        tNow + np.arange(-nSprints + 1, 1) * dt * nSteps, 6
    )
    # writing to a file when having numRec slices
    fileName = write2ncfile(timeRec, zetaRec, psiRec, x, y)
    print("File saved: " + fileName)

    timerWrite = time.time() - walltime + timerWrite
    walltime = time.time()


print("Completion of integration\n")

print("Setup:       %10.6f" % timerSetup)
print("1st Relay:   %10.6f" % timerRelay1)
print("Other %3i:   %10.6f" % (iTotal / nSprints / nSteps - 1, timerRelayO))
print("Writing Data:%10.6f" % timerWrite)
