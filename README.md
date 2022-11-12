# BVEX
Barotropicl Vorticity Equation in JAX (BVEX)

This project comprises a barotropic vorticity model coded with JAX and machine learning examples. 

The BVEX model can be run in a standalone mode and has great speed on a GPU [using `run_standalone.py` and `namelist.py`].

A few deep learning (DL) strategies are evaluated to investigate the potentials of DL parameterization of subgrid-scale (SGS) processes.

An example of creating TFDS dataset for DL training is in the `highres_forcing_long` directory.

An example of creating TFDS dataset for Transfer Learning is in the `observation_history` directory.

For DL training code -- see `DL` directory.

For TL training code -- see `TL` directory;



