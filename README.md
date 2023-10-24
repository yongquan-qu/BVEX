# BVEX
Code for the paper 
> Qu, Y., and X. Shi, 2023: Can a Machine Learning–Enabled Numerical Model Help Extend Effective Forecast Range through Consistently Trained Subgrid-Scale Models?. Artif. Intell. Earth Syst., 2, e220050, https://doi.org/10.1175/AIES-D-22-0050.1.

Barotropic Vorticity Equation in JAX (BVEX)

This project comprises a barotropic vorticity model coded with JAX and machine learning examples. 

The BVEX model can be run in a standalone mode and has great speed on a GPU [using `run_standalone.py` and `namelist.py`].

A few deep learning (DL) strategies are evaluated to investigate the potentials of DL parameterization of subgrid-scale (SGS) processes.

An example of creating TFDS dataset for DL training is in the `highres_forcing_long` directory.

An example of creating TFDS dataset for Transfer Learning is in the `observation_history` directory.

For DL training code -- see `DL` directory.

For TL training code -- see `TL` directory;



