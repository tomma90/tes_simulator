import numpy as np
import matplotlib.pyplot as plt
import tes_simulator as tessim
import pandas as pd
import os
import sys
import datetime

tesA = tessim.tes_ac_model()

sampling_rate = 20e6
time = 1

t = np.linspace(0., time, int(sampling_rate*time+1))
P = np.ones_like(t) * tesA.optical_loading_power 
Ib = np.ones_like(t) * tesA.biasing_current_amplitude
Tb = np.ones_like(t) * tesA.temperature_focal_plane
I = np.zeros_like(t)
T = np.zeros_like(t)

start = datetime.datetime.now()
I, T = tessim.TesAcRungeKuttaSolver(t, Ib, P, Tb, I, T, tesA)
finish = datetime.datetime.now()
print(finish-start)

plt.figure()
plt.plot(t, I)

plt.figure()
plt.plot(t, T)
plt.show()
