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
self.z_ch += 50.
self.updateTopo()
self.topoPlot()
self.z_ch -= 10
self.updateTopo()
self.topoPlot()
self.z_ch += 30
self.updateTopo()
self.topoPlot()
self.z_ch -= 2
self.updateTopo()

self.topoPlot('ko-')
plt.show()
