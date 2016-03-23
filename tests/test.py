#! /usr/bin/python

from terrapin import *

from matplotlib import pyplot as plt

self = Terrapin()
self.initialize()
self.topoPlot()
self.z_ch = -60.
self.updateFluvialTopo_z()
self.topoPlot()
self.z_ch -= 10.
self.updateFluvialTopo_z()
self.topoPlot()
self.z_ch += 50.
self.updateFluvialTopo_z()
self.topoPlot()
self.z_ch -= 10
self.updateFluvialTopo_z()
self.topoPlot()
self.z_ch += 20
self.updateFluvialTopo_z()
self.topoPlot()
self.z_ch -= 2
self.updateFluvialTopo_z()

self.topoPlot('ko-')
plt.show()
