# tessimdc
A code to simulate a Transition Edge Sensor (TES) response according to:
[Irwin & Hilton, Transition edge sensors, 2005](https://doi.org/10.1007/10933596_3).

We also provide an AC bias version based on the DfMux readout scheme: 
[Dobbs et al. 2012](https://doi.org/10.1063/1.4737629).

The code is a new implementation of tessimdc by the same author. All the functions have been rewritten to speed up the calculation using numba.
In order to use it import the module tes_simulator.py and give a look to test_dc.py or test_ac.py for some example on how to use the code.

The code uses the following python modules:

numpy

numba

collections
