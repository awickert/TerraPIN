#! /usr/bin/python

from terrapin import *

tpin = Terrapin()
tpin.set_input_values()
tpin.z_br_ch = -60.
tpin.incise()
tpin.z_br_ch -= 10.
tpin.incise()

