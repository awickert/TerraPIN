#! /user/bin/python

# Started 22 January 2015 by ADW
# based on some earlier notes and code

import numpy as np
from matplotlib import pyplot as plt

class Terrapin(object):
  """
  Terrapin (or TerraPIN) stands for "Terraces put into Numerics". It is a module 
  that generates the expected terraces, both strath and fill, from prescribed 
  river aggradation and degradation (incision).

  It works on a single river cross-section, and is meant to include both bedrock 
  and alluvium, and in theory should be able to contain as many layers as it 
  likes with different material properties ... with some well-organized 
  programming being the prerequisite, of course!
  """

  # HIGH-LEVEL CORE FUNCTIONS
  def __init__(self):
    pass
    
  def main(self):
    self.initialize()
    self.update() # For some time -- really just placeholder at moment
    self.finalize()

  def initialize(self):
    self.set_input_values()
    self.topographicProfile(self.layer_tops)

  def update(self):
    self.updateTopo()
    #self.erode_laterally()

  def finalize(self):
    pass

  # FUNCTIONS TO HANDLE PARTS OF THE RUNNING
  def set_input_values(self):
    """
    Eventually will set input variable values in some more end-user-interactive
    way, but currently just is where I type the values into this file itself.
    Early stages of development!
    """
    # elevation of the channel bed at the starting time
    self.z_ch = 0.
    # Angle of repose of alluvium
    alpha_a = 32.
    # Angle of repose of bedrock 
    alpha_r = 75.
    # PROBABLY PUT THESE TWO IN A LIST TO WRITE MORE GENERAL CODE THAT CAN 
    # CYCLE THROUGH ANY NUMBER OF LAYER AND THEIR PARAMETERS
    k_a = 1E-2
    # Erodibility coefficient of bedrock
    k_r = 1E-4
    # Surface elevation profiles -- start out flat
    # bedrock (x,z)
    z_br = np.array([[-np.inf, -10], [0, -10]])
    # sediment (x,z)
    z_sed = np.array([[-np.inf, 0], [0, 0]])
    # Define channel width at this point
    self.b = 50 # [m]
    # And the channel will start out the same width as its bedrock-valley
    self.B = 50 # [m]
    
    # Create arrays of values of angles and resistance to erosion
    # Alluvium is always the last one
    # Goes from bottom of strat column to top
    """
    self.alpha = [alpha_r, alpha_a]
    self.k = [k_r, k_a]
    self.layer_tops = [z_br, z_sed]
    self.layer_names = ['bedrock', 'alluvium']
    self.layer_numbers = np.arange(len(self.layer_tops))
    """
    self.alpha = {'bedrock': alpha_r, 'alluvium': alpha_a}
    self.k = {'bedrock': k_r, 'alluvium': k_a}
    self.layer_tops = [z_br, z_sed]
    self.layer_names = ['bedrock_0', 'alluvium_0']
    self.layer_numbers = np.array([0, 1])
    self.layer_lithologies = ['bedrock', 'alluvium']

  def updateTopo(self):
    self.z_ch_old = self.topo[-1,-1]
    self.dz = self.z_ch - self.z_ch_old
    if self.dz > 0:
      self.aggrade()
    elif self.dz < 0:
      self.incise()
    elif self.dz == 0:
      pass
    else:
      sys.exit("Warning: dz is not finite")

  def incise(self):
    """
    When river incises, compute erosion and angle-of-repose destruction of
    material above.
    When collapse of bedrock does happen, it creates an angle-of-repose pile
    of sediments in the real world. Represent this in some way?
    """
    point = np.array([0, self.z_ch])
    # And layer_updates holds new points that modify layers until the end,
    # when we are ready to update the whole system all at once
    layer_updates = []
    while point is not None:
      inLayer = self.insideWhichLayer(point)
      if inLayer is None:
        sys.exit("should alluviate here! improve code!")
      else:
        print "*", inLayer
        # slope-intercept
        angleOfRepose = self.alpha[self.layer_lithologies[inLayer]]
        m = - np.tan( (np.pi/180.) * angleOfRepose)
        b = point[1] - m*point[0]
        # Find intersection with each layer
        intersection = []
        for i in self.layer_numbers:
          # list of 1D numpy arrays
          intersection.append(self.findIntersection(m, b, self.layer_tops[i]))
        intersection = np.array(intersection)
        # Define the chosen intersection
        chosenintersectionion = intersection.copy()
        # First, use only those points are above the point in question
        intersection[intersection[:,1] <= point[1]] = np.nan
        # if nothing above, then we are at the top
        if np.isnan(intersection).all():
          # Break out of loop
          point = None
        else:
          path_lengths = ( (intersection[:,0] - point[0])**2 \
                         + (intersection[:,1] - point[1])**2 )
          # And of these nonzero paths, find the shortest, and this is the
          # chosen intersection
          # Chosen layer number will work here because self.layer_numbers is
          # ordered just by a np.arange (so same)
          chosen_layer_number = (path_lengths == np.nanmin(path_lengths)).nonzero()[0][0]
          chosenintersectionion = intersection[chosen_layer_number]
          layer_updates.append([chosen_layer_number, chosenintersectionion])
          # Now note that chosenintersectionion is the new starting point
          point = chosenintersectionion.copy()
    
    # Wait until the end to update the cross-sectional profile
    for i in range(len(layer_updates)):
      layer_number = layer_updates[i][0]
      intersection = layer_updates[i][1]
      # Then add this into the proper layer
      print intersection
      self.layer_tops[layer_number] = np.vstack(( self.layer_tops[layer_number], np.expand_dims(intersection, 0) ))
      
    # Sort it in order of increasing x so it is at the proper point
    for i in range(len(layer_updates)):
      self.layer_tops[i] = self.layer_tops[i][ self.layer_tops[i][:,0].argsort()]

    # And after this, adjust the right-hand-sides of the layers to hit the river
    # NOT SURE THAT THIS IS REALLY NECESSARY -- LAYER VS. TOPOGRAPHIC SURFACE
    #if self.layer_tops[0][-1,-1] > self.z_ch:
    #  self.layer_tops[0][-1,-1] = self.z_ch
      # MAY HAVE TO ADD IN A SECOND POINT HERE ONCE TERRACES / LATERAL
      # MOTION COME ON LINE
    # NEED TO DEFINE TOPOGRAPHIC SURFACE SOMEWHERE
    # CHANNEL, VALLEY WALL, AND EACH FARTHEST RIGHT POINT ON EACH LAYER TOP

    self.topographicProfile(layer_updates)
    
    # Remove points that go beyond topo profile
    layer_top_index = 0
    for layer in self.layer_tops:
      row_indices = []
      row_index = 0
      for point in layer:
        x_point = point[0]
        y_point = point[1]
        y_topo = self.piecewiseLinearAtX(x_point, self.topo)
        if y_point <= y_topo:
          pass
        else:
          print point
          row_indices.append(row_index)
        row_index += 1
      self.layer_tops[layer_top_index] = \
          np.delete(self.layer_tops[layer_top_index], row_indices, axis=0)
      layer_top_index += 1
      
    # Add points at top of layer below, to follow topography
    for i in range(1, len(self.layer_tops)):
      self.layer_tops[i] = \
          np.append(self.layer_tops[i], 
                    np.expand_dims(self.layer_tops[i-1][-1], 0),
                    axis=0)
    self.layer_tops[0] = \
      np.append(self.layer_tops[0], 
                np.expand_dims(self.topo[-1], 0),
                axis=0)
    

    # Probably unnecessary, but removing duplicates following the answer at:
    # http://stackoverflow.com/questions/8560440/
    # removing-duplicate-columns-and-rows-from-a-numpy-2d-array
    for i in range(len(layer_updates)):
      unique = np.unique(self.layer_tops[i].view([('', \
               self.layer_tops[i].dtype)] * \
               self.layer_tops[i].shape[1]))
      self.layer_tops[i] = unique.view \
                                (self.layer_tops[i].dtype) \
                                .reshape((unique.shape[0], self.layer_tops \
                                [i].shape[1]))

  def erode_laterally(self):
    pass

    for i in range(len(self.layer_tops)):
      print ""
      print self.layer_tops[i]
    print ""
    print "================="
  
  def aggrade(self):
    """
    Alluvial aggradation fills the valley with sediment up to a certain level.

    If above the valley (i.e. flat plain), (should just disperse sediment 
    over some distance. But for now will just error out.
    """
    x_valley_wall = self.piecewiseLinearAtZ(self.z_ch, self.topo)
    aggraded_surface = np.array([[x_valley_wall, self.z_ch],
                                 [0, self.z_ch]])
    # Find uppermost points below and to the right;
    # These will be surface below.
    
    # Maybe I have to bite the bullet and just make full layer polygons, or
    # at least sideways U-shapes with ends at infinity.
    
    # Geologic layers always go from top to bottom.
    # So layers below give bottom of layer above.
    # Layers below and within topo.
    alluv_layers = np.array(self.layer_lithologies) == 'alluvium'
    
    # And then see if any of this is alluvium
    # And/or see where alluvium is
    # SPACE HERE TO ADD A NEW LAYER OR INCORPORATE IT INTO OTHERS
    
  def topographicProfile(self, layers):
    # Topographic profile
    topo = []
    topo.append([0, self.z_ch])
    for row in layers:
      topo.append(list(row[1]))
    topo.append(list(self.layer_tops[-1][0]))
    self.topo = np.array(topo)[::-1]
  
  def layer_boundaries(self):
    """
    For each line segment:
      find all highest points that are below it.
      append these to a big line.
    Attach this to the layer_top stuff, as the layer bottom
    And then there is a closed boundary, and I can write methods to test what
    touches this or is inside it.
    """
  
  def linemesh(self):
    """
    Something like this is likely to take the job of "aggrade" and "incise"
    in order to build the lines in the model at each time step.
    """
    pass

  # Other functions
  def weathering(self):
    """
    Thanks to suggestion by Manfred Strecker about this; will be placeholder
    for a possible third layer that will grow in time -- loose regolith.
    """
    pass
    
  #####################  
  # Utility functions #
  #####################
  
  def findIntersection(self, m, b, piecewiseLinear):
    """
    Find intersection between two lines.
    
    m, b for angle-of-repose slope (m) and intercept (b) coming up from river,
         or, if this is after the first river--valley wall intersection,
         somewhere on the slope above the river
    
    piecewiseLinear: A piecewise linear set of (x,y) positions for the next 
                     geologic unit above the river. This should take the form:
                     ((x1,y1), (x2,y2), (x3,y3), ..., (xn, yn))
    
    """
    intersection = None
    for i in range(len(piecewiseLinear)-1):
      # Piecewise linear preparation
      # Because of sorting, xy0 is always < xy1
      xy0 = piecewiseLinear[i]
      xy1 = piecewiseLinear[i+1]
      # slope and intercept
      if np.isinf(xy1[1]) or np.isinf(xy0[1]):
        m_pwl = np.nan
        b_pwl = np.nan
      else:
        m_pwl = (xy1[1] - xy0[1])/(xy1[0]-xy0[0])
        b_pwl = xy1[1] - m_pwl * xy1[0] # use xy1 to avoid inf
      # Then find the intersection with the line
      # x-value first
      xint = (b_pwl - b)/(m - m_pwl)
      # Next, see if the point exists
      # >= and <= to end on the first one it hits
      if (xint >= xy0[0]) and (xint <= xy1[0]):
        # You're done!
        # y-value plugging into one equations -- let's use the line from the 
        # starting point that is producing the erosion
        yint = m*xint + b
        # Because there is some numerical error created by defining yint
        # with such an equation let's add a rounding term: round to nearest
        # 1E-9 m (nanometer) -- because totally insignificant in these systems
        intersection = np.round(np.array([xint, yint]), 9)
        break
    # If at the end of the loop, nothing has been found, 
    # replace it with np.nan
    if intersection is not None:
      pass
    else:
      intersection = np.array([np.nan, np.nan])

    return intersection
  
  def piecewiseLinearAtX(self, x, pwl):
    """
    Evaluates a piecewise linear expression to solve for z at a given x.
    x:   the x-value
    pwl: a 2-column numpy array, ([x, z]), that must contain at least
         four entries (two points define a line)
    """
    
    # First check if z(x) not defined for this line.
    if x < np.min(pwl[:,0]) or x > np.max(pwl[:,0]):
      z = np.nan
    else:
      # First, define line segment of interest
      xmin_pwl = np.max( pwl[:,0][pwl[:,0] <= x] )
      xmax_pwl = np.min( pwl[:,0][pwl[:,0] >= x] )
      z_xmin_pwl = float(pwl[:,1][pwl[:,0] == xmin_pwl])
      z_xmax_pwl = float(pwl[:,1][pwl[:,0] == xmax_pwl])
      if xmin_pwl == xmax_pwl:
        z = z_xmin_pwl
      else:
        # z = mx + b 
        m = (z_xmax_pwl - z_xmin_pwl)/(xmax_pwl - xmin_pwl)
        # using max to avoid -inf
        b = z_xmax_pwl - m*xmax_pwl
        z = m*x + b
      
    return z
    
  def piecewiseLinearAtZ(self, z, pwl):
    """
    Evaluates a piecewise linear expression to solve for x at a given z.
    z:   the z-value
    pwl: a 2-column numpy array, ([x, z]), that must contain at least
         four entries (two points define a line)
    """
    
    # First check if x(z) not defined for this line.
    if z < np.min(pwl[:,0]) or z > np.max(pwl[:,0]):
      z = np.nan
    else:
      # First, define line segment of interest
      zmin_pwl = np.max( pwl[:,1][pwl[:,1] <= z] )
      zmax_pwl = np.min( pwl[:,1][pwl[:,1] >= z] )
      # Using np.max here because not necessarily a function in (z, x) space
      # (horizontal lines); max value matters for alluviation
      x_zmin_pwl = float(np.max(pwl[:,0][pwl[:,1] == zmin_pwl]))
      x_zmax_pwl = float(np.max(pwl[:,0][pwl[:,1] == zmax_pwl]))
      if zmin_pwl == zmax_pwl:
        x = x_zmin_pwl
      else:
        # z = mx + b 
        m = (x_zmax_pwl - x_zmin_pwl)/(zmax_pwl - zmin_pwl)
        # using max to avoid -inf
        bz = x_zmax_pwl - m*zmax_pwl
        x = m*z + bz
      
    return x
 
  def nextToWhichLayer(self, point):
    pass
  
  def insideWhichLayer(self, point):
    """
    Point is (x,z)
    This script will return which layer the point is in.
    Because it is used to find which angle is the proper angle of repose
    for the rock/sediment/soil/etc. above, if it is on the border between
    two units, it will pick the upper one.
    
    The importance of this script lies in the fact that each time the river 
    incises, the point will end up in the ground, and will have to project out 
    of that. In plausible cases, it might end up projecting back up through 
    multiple materials.
    """
    
    layer_elevations_at_point = []
    for i in range(len(self.layer_tops)):
      layer_elevations_at_point.append(self.piecewiseLinearAtX(point[0], self.layer_tops[i]))
    layer_elevations_at_point = np.array(layer_elevations_at_point)

    # Find lowest elevation above point
    print layer_elevations_at_point, point
    # Get invalid value error if there is a nan, which means that the layer
    # does not exist above that point
    # But this will always be false anyway, so this is fine. Just suppress
    # the error
    with np.errstate(invalid='ignore'):
      layers_above_point = layer_elevations_at_point > point[1]
    if layers_above_point.any():
      layer_elevation_point_is_inside = \
        np.min(layer_elevations_at_point[layers_above_point])
    # while I see the main use of this as checking for incision, thereby 
    # making these next statements not needed, I will check if the point is
    # at or above the highest layer
    # MAYBE CUTOFF HERE WITH OPTION SO WE DON'T RETURN LAYER TOPS WHEN YOU REALLY
    # WANT TO KNOW WHAT YOU'RE INSIDE, ONLY.
    else:
      layers_at_or_above_point = layer_elevations_at_point >= point[1]
      if layers_at_or_above_point.any():
        layer_elevation_point_is_inside = layer_elevations_at_point[layers_at_or_above_point]
      else:
        # If neither of these, must be above everything
        layer_elevation_point_is_inside = None
        layer_number = None
      
    if layer_elevation_point_is_inside is not None:
      layer_number = self.layer_numbers[layer_elevations_at_point == layer_elevation_point_is_inside]
      layer_number = int(layer_number)
    
    return layer_number
    
  def store_layers(self):
    """
    Save layers for visualization and analysis
    """
    pass

  """
    dz_ch__dt = -2.
    
    for t in [1]:
      z_ch += dz_ch__dt
      # In what material does the base of the channel lie?
      # Easy since channel is at 0
      # And what are all the layers above?
      layer_bases = []
      # Going to assume for now that the layers always stay in the same order
      # and go from the bottom to the top
      for layer in lith:
        layer_bases.append(layer[-1,-1]) # Fix this to calculate those 
                                         # values at chosen x
                                         # can write a function to do this
         # Then can find those above the point that we care about.
         # But what if one below has a steep slope?! This won't work or matter!
         # Just see what unit you are in and find the connection along this line
         # with the next one up
      # Build a line up from z_ch
      for i in range(len(lith)):
        for segment in range(len(lith[i]-1)):
          intercept = 
      # Ignore sediments for now
      
      
      intersection
      
      
      
      
      
      
      
      
    xo = origin[0]
    zo = origin[1]
    xi = intersection[0]
    zi = intersection[1]
    

  """    
      

    
    
    
    
