#! /usr/bin/python

import terrapin

reload(terrapin)

from matplotlib import pyplot as plt

self = terrapin.Terrapin()
self.initialize()
self.topoPlot()
self.z_ch = -60.
self.updateFluvialTopo_z()
self.topoPlot()
self.z_ch -= 1.
self.updateFluvialTopo_z()
self.updateFluvialTopo_z()
self.topoPlot()
self.z_ch -= 1.
self.updateFluvialTopo_z()

self.topoPlot()
self.z_ch -= 10.
self.updateFluvialTopo_z()

self.topoPlot()
self.z_ch += 15
self.updateFluvialTopo_z()

self.topoPlot()
self.z_ch -= 1



self.updateFluvialTopo_z()

self.topoPlot()
"""
self.z_ch -= 10
self.updateFluvialTopo_z()
self.topoPlot()

self.z_ch += 20
self.updateFluvialTopo_z()
self.topoPlot()

self.z_ch -= 2
self.updateFluvialTopo_z()
self.topoPlot()

self.z_ch -= 2
self.updateFluvialTopo_z()
self.topoPlot()
self.z_ch -= 2
self.updateFluvialTopo_z()
self.topoPlot()
self.z_ch -= 10
self.updateFluvialTopo_z()
"""

self.topoPlot('ko-')
plt.show()
