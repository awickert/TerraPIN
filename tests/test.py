#! /usr/bin/python

from terrapin import *

from matplotlib import pyplot as plt

self = Terrapin()
self.initialize()
self.z_ch = -60.
self.updateTopo()
plt.plot(self.topo[:,0], self.topo[:,1])
self.z_ch -= 10.
self.updateTopo()
plt.plot(self.topo[:,0], self.topo[:,1])
self.z_ch -= 20.
self.updateTopo()
plt.plot(self.topo[:,0], self.topo[:,1])

plt.show()
