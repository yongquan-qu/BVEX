# This script is an older standalone verion of BVEX. It is included here for 
# demonstrating the chaos in this system. If you compare results from this
# code and the current code, you will find that at t=30 notable differences
# appear, and by t=50 they look significantly different. 
#     This code carries wave coefficient as model state while the lastest 
# version carries vorcticity in physical space.


import jax
import jax.numpy as jnp
from jax.numpy.fft import fft2, ifft2
import numpy.fft as nfft
import netCDF4 as nc4
from namelist import *

import time


# timers for performance evaluation
timerSetup = 0.0
timerRelay1 = 0.0  # first relay
timerRelayO = 0.0  # other relays
timerWrite = 0.0
walltime = time.time()


def setup_ic_grid(shear_frac, rand_seed):
    """ Setup initial conditions """

    x1d = np.linspace(2.0 * np.pi / nx / 2, 2.0 * np.pi - 2.0 * np.pi / nx / 2, nx)
    y1d = np.linspace(2.0 * np.pi / ny / 2, 2.0 * np.pi - 2.0 * np.pi / ny / 2, ny)
    x2d, y2d = np.meshgrid(x1d, y1d, indexing="ij")

    tmp1 = np.linspace(0, nx / 2 - 1, int(np.around(nx / 2))) * 2 * np.pi / lx
    tmp2 = np.linspace(-nx / 2, -1, int(np.around(nx / 2))) * 2 * np.pi / lx
    kx1d = np.concatenate((tmp1, tmp2))
    tmp1 = np.linspace(0, ny / 2 - 1, int(np.around(ny / 2))) * 2 * np.pi / ly
    tmp2 = np.linspace(-ny / 2, -1, int(np.around(ny / 2))) * 2 * np.pi / ly
    ky1d = np.concatenate((tmp1, tmp2))
    kx2d, ky2d = np.meshgrid(kx1d, ky1d, indexing="ij")
    k2 = kx2d * kx2d + ky2d * ky2d
    del2 = -k2
    del2[0, 0] = 1

    # #---------------------------
    # # Purely random noise; for Komogorov flow or others                       
    # q_0 = np.zeros((nx, ny))                                                
    # q0_h = fft2(q_0)                                                        
    # p0_h = q0_h / del2                                                      
    # p_0 = np.real(ifft2(p0_h))                                              
    # np.random.seed(rand_seed)                                               
    # q_0 = q_0 + (np.random.uniform(0.0, 1.0, (nx, ny)) - 0.5) / 10.0  
    # #---------------------------

    #---------------------------
    # Shear zone setup for simpler Kelvin-Helmholtz instability
    q_0 = -(shear_frac / (2 * shear_frac - 1)) * np.sin(
        np.pi - (2 * shear_frac / (2 * shear_frac - 1) * y2d)
    )
    q_0[
        (y2d >= (2 * shear_frac - 1) / (2 * shear_frac) * np.pi)
        & (y2d <= (2 * shear_frac + 1) / (2 * shear_frac) * np.pi)
    ] = (2 * shear_frac / np.pi)
    q_0[y2d > (2 * shear_frac + 1) * np.pi / (2 * shear_frac)] = -(
        shear_frac / (2 * shear_frac - 1)
    ) * np.sin(
        (2 * shear_frac)
        / (2 * shear_frac - 1)
        * y2d[y2d > (2 * shear_frac + 1) * np.pi / (2 * shear_frac)]
        - (2 * shear_frac + 1) / (2 * shear_frac - 1) * np.pi
    )
    q0_h = fft2(q_0)
    p0_h = q0_h / del2
    p_0 = np.real(ifft2(p0_h))
    np.random.seed(rand_seed)
    q_0 = q_0 + (np.random.uniform(0.0, 1.0, (nx, ny)) - 0.5) / 10.0 * np.max(
        np.abs(q_0)
    )
    #---------------------------

    return q_0, p_0, kx2d, ky2d, del2, x2d, y2d, x1d, y1d


def rhs_tendency(q_hat, t_now, del2, kx2d, ky2d, p0, y2d):
    """ Compute the RHS tendency due to advection and forcing """
    p_hat = q_hat / del2
    q = jnp.real(ifft2(q_hat))
    dpdx = jnp.real(ifft2(1j * kx2d * p_hat))
    dpdy = jnp.real(ifft2(1j * ky2d * p_hat))
    dqdx = jnp.real(ifft2(1j * kx2d * q_hat))
    dqdy = jnp.real(ifft2(1j * ky2d * q_hat))

    adv = dpdx * dqdy - dpdy * dqdx
    adv_hat = fft2(adv)
    
    # #---------------------------
    # # Komogorov forcing
    # u_forcing = np.sin(4.0*y2d) 
    # forcing_hat = -1j * ky2d * nfft.fft2(u_forcing)
    # #---------------------------

    #---------------------------
    # Simple shear zone forcing
    p_now = jnp.real(ifft2(p_hat))
    forcing = (
        -alpha
        * (p_now - p0)
        * ((1.0 - jnp.cos(t_now / 5.0 * jnp.pi)) / 2.0) ** 4.0
        * (((1.0 - jnp.cos(y2d)) / 2.0) ** 4 + 1.0 / 24.0)
        * (24.0 / 25.0)
    )
    forcing_hat = del2 * fft2(forcing)
    #---------------------------

    rhs = -adv_hat + forcing_hat

    return rhs


@jax.jit
def edtrk4(q_hat, t_now):
    """ Integrate the model for one step with the EDTRK4 scheme """
    # https://github.com/navidcy/barotropic_QG

    # Define model parameters as numpy arrays so that they become static later
    # obtain p0 and wavenumber info from setupICs
    _, p0, kx2d, ky2d, del2, x2d, y2d, _, _ = setup_ic_grid(shearFrac, randSeed)

    k2 = kx2d * kx2d + ky2d * ky2d

    s = 4
    k = np.sqrt(k2)
    k_max = ny / 2
    k_max_s = k_max * (lx / nx)
    k_cut = 2 / 3 * k_max
    k_cut_s = k_cut * (ly / ny)
    a = -np.log(1e-15) / ((k_max_s - k_cut_s) ** s) * ((ly / ny) ** s)
    mask = np.ones((nx, ny)) * np.abs(k <= ny / 3) + np.exp(
        -a * (k - k_cut) ** s
    ) * np.abs(k > ny / 3)
    # Calculate coefficients for the EDTRK4 algorithm """
    lin = -mu - nu * k2 ** 2
    e_lin = np.exp(lin * dt)
    e_lin2 = np.exp(lin * dt / 2)
    m_pts = 64
    r = np.exp(2j * np.pi / m_pts * np.linspace(1, m_pts, m_pts))
    fu = np.zeros((nx, ny))
    fab = fu
    fc = fu
    h_2 = fu  # in the limit lin->0, this coefficient becomes dt/2
    for m in range(0, m_pts):
        z = r[m] + lin * dt
        h_2 = h_2 + dt * (np.exp(z / 2) - 1) / z
        fu = fu + dt * (-4 - z + np.exp(z) * (4 - 3 * z + z ** 2)) / z ** 3
        fab = fab + dt * (+2 + z + np.exp(z) * (-2 + z)) / z ** 3
        fc = fc + dt * (-4 - 3 * z - z ** 2 + np.exp(z) * (4 - z)) / z ** 3
    fu = fu / m_pts
    fab = fab / m_pts
    fc = fc / m_pts
    h_2 = h_2 / m_pts

    # integration for one step
    nlin0_z = rhs_tendency(q_hat, t_now, del2, kx2d, ky2d, p0, y2d)
    k1z = e_lin2 * q_hat + h_2 * nlin0_z
    nlin1_z = rhs_tendency(k1z, t_now, del2, kx2d, ky2d, p0, y2d)
    k2z = e_lin2 * q_hat + h_2 * nlin1_z
    nlin2_z = rhs_tendency(k2z, t_now, del2, kx2d, ky2d, p0, y2d)
    k3z = e_lin2 * k1z + h_2 * (2 * nlin2_z - nlin0_z)
    nlin3_z = rhs_tendency(k3z, t_now, del2, kx2d, ky2d, p0, y2d)

    q_h_new = (
        e_lin * q_hat + fu * nlin0_z + 2 * fab * (nlin1_z + nlin2_z) + fc * nlin3_z
    )

    q_h_new = q_h_new * mask

    t_new = jnp.around(t_now + dt, 6)
    return q_h_new, t_new


@jax.jit
def sprint(physics_state, _):
    """ Integrate the model for nSteps steps """

    q_hat, t_now = jnp.split(physics_state, 2, axis=0)
    t_set = jnp.real(t_now[0, 0, 0]) + np.arange(nSteps) * dt

    q_hat_new, t_set_new = jax.lax.scan(edtrk4, jnp.squeeze(q_hat), t_set)

    t_new = jnp.around(jnp.full((nx, ny), t_set_new[-1]), 6)
    physics_state_new = jnp.stack((q_hat_new, t_new), axis=0)
    # both outputs have q_hat_new, but only the second will be stacked if this
    # function is scanned by another function
    return physics_state_new, q_hat_new


@jax.jit
def relay(_, physics_state):
    """ Run a number of sprints to obtain n_sprints output slices 

    physics_state is the I.C. for a given time. The function integrates for 
    nSprints*nSteps*dt time and saves nSprints output slices.

    ``carry`` (_) is a dummy argument for now. 
 
    In standalone integration, physics_state_end can be used for the next loop
    (using while or for).
    """

    # Integrate the model for nSprints
    physics_state_end, q_hat_set = jax.lax.scan(
        sprint, physics_state, None, length=nSprints
    )
    return physics_state_end, q_hat_set


@jax.jit
def race(physics_state_batch):
    """ Apply ``relay`` to a batch of physics states

    Output q_hat_batch[batch, time_slices, x, y]
    """

    # Integrate the batch for nSprints each
    carry = physics_state_batch[0]
    _, q_hat_batch = jax.lax.scan(relay, carry, physics_state_batch)

    return q_hat_batch


def write2ncfile(time_rec, zeta_rec, psi_rec, ix, iy):
    """ Write data to a NetCDF file """
    start_time = time_rec[0]
    end_time = time_rec[-1]
    filename = fileNameFormat % (start_time, end_time)
    ncfile = nc4.Dataset(filename, mode="w", format="NETCDF4")
    ncfile.createDimension("y", ny)
    ncfile.createDimension("x", nx)
    ncfile.createDimension("time", None)

    y_nc = ncfile.createVariable("y", np.float32, ("y",))
    y_nc.long_name = "y"
    x_nc = ncfile.createVariable("x", np.float32, ("x",))
    x_nc.long_name = "x"
    itime = ncfile.createVariable("time", np.float32, ("time",))
    itime.long_name = "time"

    zeta = ncfile.createVariable(
        "zeta", np.float32, ("time", "y", "x"), fill_value=1.0e36
    )
    zeta.long_name = "vorticity"

    psi = ncfile.createVariable(
        "psi", np.float32, ("time", "y", "x"), fill_value=1.0e36
    )
    psi.standard_name = "streamfunction"

    x_nc[:] = ix
    y_nc[:] = iy
    zeta[:, :, :] = zeta_rec[:, :, :]
    psi[:, :, :] = psi_rec[:, :, :]
    itime[:] = time_rec[:]

    ncfile.close()
    return filename


#=====================================================
# Main body

# Create array to save data (for ML training these need to be stacked/concatenated jnp arrays)
numRec = np.min([1000, nSprints]).astype(int)
zetaRec = np.empty((numRec, ny, nx), dtype=np.float32)
psiRec = np.empty((numRec, ny, nx), dtype=np.float32)
timeRec = np.empty((numRec,), dtype=np.float32)

timerSetup = time.time() - walltime + timerSetup
walltime = time.time()

# Start the integration
timerSetup = time.time() - walltime + timerSetup
walltime = time.time()

q0, _, Kx, Ky, Del2, _, _, x, y = setup_ic_grid(shearFrac, randSeed)

if isRestart: 
    dsFile = nc4.Dataset(restartFile)
    zeta = np.copy(dsFile.variables["zeta"]).astype("float32")
    q0 = np.transpose(zeta[-1]).astype("float32")
    ncTime = np.copy(dsFile.variables["time"]).astype("float32")
    qHat = fft2(q0) 
    time2d = np.zeros((nx, ny), dtype=np.complex64) + ncTime[-1].astype(np.complex64)
else:
    qHat = fft2(q0) 
    time2d = np.zeros((nx, ny), dtype=np.complex64) 


physicsState = jnp.stack((qHat, time2d), axis=0)

it = 0
iTotal = np.around(t_max / dt).astype(int)


while it < iTotal:
    physicsState, qHatSet = relay(None, physicsState)

    if it < (nSteps * nSprints - 1):
        timerRelay1 = time.time() - walltime + timerRelay1
        walltime = time.time()
    else:
        timerRelayO = time.time() - walltime + timerRelayO
        walltime = time.time()

    it = it + nSteps * nSprints

    # record data
    q = np.real(nfft.ifft2(qHatSet))
    pHatSet = qHatSet / Del2
    p = np.real(nfft.ifft2(pHatSet))
    u = np.real(nfft.ifft2(-1j * Ky * pHatSet))
    v = np.real(nfft.ifft2(+1j * Kx * pHatSet))
    cfl = np.sqrt(np.max(u ** 2 + v ** 2)) * dt / dx
    print(("Relay #%4i;  CFL =%5.2f " % (int(it / nSteps / nSprints), cfl)))
    zetaRec = np.transpose(q, axes=[0, 2, 1]).astype("float32")
    psiRec = np.transpose(p, axes=[0, 2, 1]).astype("float32")
    _, time2d = physicsState
    timeRec = np.real(time2d[0, 0]) + np.arange(-nSprints + 1, 1) * dt * nSteps
    # writing to a file when having numRec slices
    fileName = write2ncfile(timeRec, zetaRec, psiRec, x, y)
    print("File saved: " + fileName)

    timerWrite = time.time() - walltime + timerWrite
    walltime = time.time()


print("Successful completion of integration\n")


print("Completion of integration\n")

print("Setup:       %10.6f" % timerSetup)
print("1st Relay:   %10.6f" % timerRelay1)
print("Other %3i:   %10.6f" % (iTotal / nSprints / nSteps - 1, timerRelayO))
print("Writing Data:%10.6f" % timerWrite)

