from terrapin import *

self = Terrapin()
self.set_input_values()
dt = 500
t = np.arange(0, 2000, dt)
for ts in t:
  if ts < 10000:
    dz_ch__dt = -.01 # 1 cm/yr incision
  else:
    dz_ch__dt = 0.
  self.incise()
  self.z_br_ch += dz_ch__dt * dt



"""
point = np.array([0, self.z_br_ch])
angleOfRepose = self.alpha[0]
# Slope -- minus because solving for what is left of river
m = - np.tan( (np.pi/180.) * angleOfRepose)
# Intercept
b = point[1] - m*point[0]

piecewiseLinear = self.layer_tops[0]
"""
