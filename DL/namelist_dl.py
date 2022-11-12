""" This module contains the namelist variables for the barotropic vorticity model """
import numpy as np

# Set up time and space domain and other parameters
# -----------------------------------------------------------------------------------------∞∞
# number of grid points
nx = 64
ny = 64
# time step; case dependent
dt = 0.05
#to aviod roundoff error cumulation
t_around = 6
# domain size
lx = 2 * np.pi
ly = lx
# grid spacing
dx = lx / nx
dy = ly / ny
# hyper-diffusivity
nu = 0.0
# linear drag; needed for Komogorov flow; e.g. 0.1 
mu = 0.0
# forcing rate (inverse of forcing time scale)
alpha = 0.5
# middle shear zone width as a fraction (e.g. 32 means a width of pi/32)
shearFrac = 32
# random seed for noise in initial condition
randSeed = 2021
# output file name
fileNameFormat = "BVEshearForcing" + str(nx) + "_ScanRelay_%0.4i-%0.4i.nc"
# total integration time; useful for standalone mode only
t_max = 100.0
# is this a restart?
isRestart = False
# restart file name
restartFile = f'/workspace/yquai/BVEX/rstfiles/res_{int(1024)}_end_{int(10)}_randSeed_{int(randSeed)}.nc'
# "sprint" time interval, i.e., save one slice every dtSprint time
dtSprint = 0.05
nSteps = np.around(dtSprint / dt).astype(int)
nSprints = 1000
#nSprints = np.around(t_max/dtSprint).astype(int)
# can be any number of sprints you want to include for a relay, but be aware of memory use
n_look_ahead = 1
