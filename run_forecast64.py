"""
This script runs the BVEX model in a forecast mode.

Key parameters are specified in namelist.py. One complete run is split into
a number of relays, which are comprised by sprints. One time slice is saved
after one sprint (multiple time steps), and one relay outputs one NetCDF file
with multiple time slices.

In forecast mode, we read a batch of ICs and make forecast, when memory allows
we run the entire batch with ``race'' to obtain all forecast at once
"""

import numpy.fft as nfft
from bvex import *

# read truth data and select ICs
dsFile = nc4.Dataset("catCoarseTruth64.nc")    
# dsFile here should be from a benchmark simulation coasened to the forecast resolution
zeta0 = np.copy(dsFile.variables["zeta"]).astype("float32")
zetaIC0 = zeta0[4049:9000:50, :, :]
# number of slices here should match 'raceBatch' size
del zeta0
qIC = np.transpose(zetaIC0, (0, 2, 1))  # transpose data from cartesian to matrix indexing
time0 = np.copy(dsFile.variables["time"]).astype("float32")
tIC = time0[4049:9000:50]
# number of times matches 'raceBatch' size
del time0

######################
# Main body for the model integration and data saving
_, _, _, _, _, _, _, x, y = setup_ic_grid(shearFrac, randSeed)


# Create array to save data (for ML training these need to be stacked/concatenated jnp arrays)
numRec = np.min([1000, nSprints + 1]).astype(int)    # 1 extra slice to include IC
zetaRec = np.empty((raceBatch, numRec, ny, nx), dtype=np.float32)
timeRec = np.empty((raceBatch, numRec), dtype=np.float32)

zetaRec[:, 0, :, :] = zetaIC0   # save initial conditions

_, tEndBatch, qHistBatch = race(qIC, tIC)

# record data
zetaRec[:, 1:, :, :] = np.transpose(qHistBatch, axes=[0, 1, 3, 2]).astype("float32")
timeRec[:, :] = tEndBatch.reshape(raceBatch, 1) + np.around(
    np.reshape(np.arange(-nSprints, 1) * dt * nSteps, (1, nSprints+1)), 6
)
# writing to a file when having numRec slices
fileName = writeBatch2ncfile(timeRec, zetaRec, x, y)
print("File saved: " + fileName)

print("Completion of integration\n")
