#! /usr/bin/python

import terrapin

import numpy as np
from matplotlib import pyplot as plt

self = terrapin.Terrapin()
self.initialize()

nts = 110
nswitch = 100
timesteps = np.arange(0, nts)
noise = 2*np.random.random_sample(len(timesteps))-1 # Random on [-1, 1)
for ts in range(nts):
  if ts < nswitch:
    dz_ch__dt = -.1 * (2 + noise[ts])
  else:
    #dz_ch__dt = 0.
    dz_ch__dt = 1 * (.5 + noise[ts])
  self.z_ch += dz_ch__dt
  self.updateFluvialTopo_z()
  # Widening ~ 1/z --> how much stuff to move
  #self.updateFluvialTopo_y( np.abs(1/self.topo[-1,-1]) )

self.layerPlot()

plt.show()
