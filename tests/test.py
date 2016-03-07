#! /usr/bin/python

from terrapin import *

from matplotlib import pyplot as plt

self = Terrapin()
self.initialize()
self.topoPlot()
self.z_ch = -60.
self.updateTopo()
self.topoPlot()
self.z_ch -= 10.
self.updateTopo()
self.topoPlot()
self.z_ch -= 20.
self.updateTopo()
self.topoPlot()

plt.show()
