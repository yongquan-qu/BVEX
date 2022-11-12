"""
This module integrates a barotropic vorticity equation with a periodic forcing.
A strong shear zone is periodically re-established to allow the development of
Kelvin-Helmholtz instability.

This module is intended for both being used in ML training and run in a
standalone model. The "relay" function accepts a model state and integrates
forward to give a certain number of output slides. It can be scanned in  ``race``
to give the output for a batch of model states.
"""

import jax
import jax.numpy as jnp
from jax.numpy.fft import fft2, ifft2
import netCDF4 as nc4

from namelist_dl import *


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


_, p0, kx2d, ky2d, del2, x2d, y2d, _, _ = setup_ic_grid(shearFrac, randSeed)


@jax.jit
def advection(q_hat, p_hat, forcing_hat, turb_hat, kx2d, ky2d):
    """ Compute the RHS tendency due to advection
    
    Forcing and SGS turbulence are computed outside and kept as constant 
    """

    dpdx = jnp.real(ifft2(1j * kx2d * p_hat))
    dpdy = jnp.real(ifft2(1j * ky2d * p_hat))
    dqdx = jnp.real(ifft2(1j * kx2d * q_hat))
    dqdy = jnp.real(ifft2(1j * ky2d * q_hat))

    adv = -(dpdx * dqdy - dpdy * dqdx)
    adv_hat = fft2(adv)

    rhs = adv_hat + forcing_hat + turb_hat

    return rhs

@jax.jit
def cal_forcing(p_now, t_now):
    
    forcing = (
        -alpha
        * (p_now - p0)
        * ((1.0 - jnp.cos(t_now / 5.0 * jnp.pi)) / 2.0) ** 4.0
        * (((1.0 - jnp.cos(y2d)) / 2.0) ** 4 + 1.0 / 24.0)
        * (24.0 / 25.0)
    )
    return forcing


@jax.jit
def laplacian(q_now):
    
    q_hat = fft2(q_now)
    p_hat = q_hat / del2
    p_now = jnp.real(ifft2(p_hat))
    
    return p_now, q_hat, p_hat
    
    
@jax.jit
def etdrk4(q_now, t_now, forcing_now=None):
    """ Integrate the model for one step with the EDTRK4 scheme """
    # https://github.com/navidcy/barotropic_QG

    # --------------------------
    # Define model parameters as numpy arrays so that they become static later
    # obtain p0 and wavenumber info from setupICs
    _, p0, kx2d, ky2d, del2, x2d, y2d, _, _ = setup_ic_grid(shearFrac, randSeed)
    k2 = kx2d * kx2d + ky2d * ky2d

    # set up filter
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
    fab = np.zeros((nx, ny))
    fc = np.zeros((nx, ny))
    h_2 = np.zeros((nx, ny))  # in the limit lin->0, this coefficient becomes dt/2
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

    # ----------------------------
    # Computation Part
    # Compute forcing

    # Simple shear-zone forcing
    
    p_now, q_hat, p_hat = laplacian(q_now)
    # input forcing or calculate forcing
    
    if forcing_now is not None:
        forcing = forcing_now
    else:
        forcing = cal_forcing(p_now, t_now)
        
    forcing_hat = del2 * fft2(forcing)
    #---------------------------
    
    # Compute SGS turbulence
    # turb = sgs_model(q_now, *args)
    # if sgs_model is defined

    turb = jnp.zeros((nx, ny))
    # zero for no-sgs_model run; note that hyper diffusion is included in edtrk4,
    # not in the SGS model

    turb_hat = del2 * fft2(turb)

    # Integration for one EDTRK4 step
    nlin0_z = advection(q_hat, p_hat, forcing_hat, turb_hat, kx2d, ky2d)
    k1z = e_lin2 * q_hat + h_2 * nlin0_z

    psi_hat = k1z / del2
    nlin1_z = advection(k1z, psi_hat, forcing_hat, turb_hat, kx2d, ky2d)
    k2z = e_lin2 * q_hat + h_2 * nlin1_z

    psi_hat = k2z / del2
    nlin2_z = advection(k2z, psi_hat, forcing_hat, turb_hat, kx2d, ky2d)
    k3z = e_lin2 * k1z + h_2 * (2 * nlin2_z - nlin0_z)

    psi_hat = k3z / del2
    nlin3_z = advection(k3z, psi_hat, forcing_hat, turb_hat, kx2d, ky2d)

    q_h_new = (
        e_lin * q_hat + fu * nlin0_z + 2 * fab * (nlin1_z + nlin2_z) + fc * nlin3_z
    )

    # Filter
    q_h_new = q_h_new * mask

    # Convert to physical space
    q_new = jnp.real(ifft2(q_h_new))

    t_new = jnp.around(t_now + dt, 6)  # avoid round-off error accumulation
    return q_new, t_new


@jax.jit
def sprint(q_now, t_now):
    """ Integrate the model for nSteps steps """

    t_set = t_now + np.arange(nSteps) * dt

    q_new, _ = jax.lax.scan(etdrk4, q_now,  t_set)

    # Two outputs are the same, but only the second will be stacked
    # if this function is scanned by another function
    return q_new, q_new


@jax.jit
def relay(q_now, t_now):
    """ Run a number of sprints to obtain n_sprints output slices 

    physics_state below is the I.C. for a given time. The function integrates for 
    nSprints*nSteps*dt time and saves nSprints output slices.
    """
    t_set = jnp.around(t_now + np.arange(nSprints) * nSteps * dt, 6)
    # around used to avoid round-off error accumulation

    # Integrate the model for nSprints
    q_end, q_hist = jax.lax.scan(sprint, q_now, t_set)

    t_end = jnp.around(t_set[-1] + nSteps * dt, 6)

    return q_end, t_end, q_hist


@jax.jit
def race(q_batch, t_batch):
    """ Apply ``relay`` to a batch of physics states
    Output q_hist_batch[batch, time_slices, x, y]
    """
    # Integrate the batch for nSprints each
    q_end_batch, t_end_batch, q_hist_batch = jax.vmap(relay)(q_batch, t_batch)
    return q_end_batch, t_end_batch, q_hist_batch


def write2ncfile(time_rec, zeta_rec, psi_rec, ix, iy):
    """ Write data to a NetCDF file """
    start_time = np.around(time_rec[0])
    end_time = np.around(time_rec[-1])
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


def write2npyfile(time_rec, zeta_rec, psi_rec, ix, iy):
    """ Write data to a .npy file """
    return None


    
vetdrk4 = jax.vmap(etdrk4)