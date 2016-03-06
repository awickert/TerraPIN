#! /usr/bin/python

from terrapin import *

from matplotlib import pyplot as plt

self = Terrapin()
self.set_input_values()
self.z_br_ch = -60.
self.incise()
plt.plot(self.topo[:,0], self.topo[:,1])
self.z_br_ch -= 10.
self.incise()
plt.plot(self.topo[:,0], self.topo[:,1])
self.z_br_ch -= 50.
self.incise()
plt.plot(self.topo[:,0], self.topo[:,1])

plt.show()
