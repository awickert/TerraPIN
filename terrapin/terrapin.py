#! /user/bin/python

# Started 22 January 2015 by ADW
# based on some earlier notes and code

import numpy as np
from matplotlib import pyplot as plt
import sys
import fnmatch
import copy

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
    self.initial_layer_boundaries = self.calc_layer_boundaries()
    self.layer_boundaries = self.initial_layer_boundaries[:]
    self.initial_layer_bottoms = self.calc_layer_bottoms()

  def update(self):
    self.layer_tops_old = self.layer_tops.copy()
    self.updateFluvialTopo_z()
    self.updateFluvialTopo_y()
    #self.erode_laterally()

  def finalize(self):
    plt.show()

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
    # Angle of repose of colluvium
    alpha_c = 20.
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
    # z_sky: try including a free space layer to make it easier
    # Define channel width at this point
    self.b = 50 # [m]
    # And the channel will start out the same width as its bedrock-valley
    self.B = 50 # [m]
    
    """
    # when erosion intersects the free air
    z_sky = np.array([[-np.inf, np.inf], [0, np.inf]])
    self.alpha = {'free_space': 0., 'bedrock': alpha_r, 'alluvium': alpha_a}
    self.k = {'free_space': np.inf, 'bedrock': k_r, 'alluvium': k_a}
    self.layer_tops = [z_sky, z_br, z_sed]
    self.layer_names = ['sky', 'bedrock_0', 'alluvium_0']
    self.layer_numbers = np.array([0, 1, 2])
    self.layer_lithologies = ['air', 'bedrock', 'alluvium']
    """
    
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
    self.alpha = {'bedrock': alpha_r, 
                  'alluvium': alpha_a,
                  'colluvium': alpha_c}
    self.k = {'bedrock': k_r, 'alluvium': k_a}
    self.layer_tops = [z_br, z_sed]
    self.layer_names = ['bedrock_0', 'alluvium_0']
    self.layer_numbers = np.array([0, 1])
    self.layer_lithologies = ['bedrock', 'alluvium']

    """
    z_br_1 = np.array([[-np.inf, -40], [0, -40]])
    self.alpha = {'bedrock_0': alpha_r, 'bedrock_1': 45, 'alluvium': alpha_a}
    self.k = {'bedrock_0': k_r, 'bedrock_1': k_r, 'alluvium': k_a}
    self.layer_tops = [z_br, z_br_1, z_sed]
    self.layer_names = ['bedrock_0', 'bedrock_1', 'alluvium_0']
    self.layer_lithologies = ['bedrock_0', 'bedrock_1', 'alluvium']
    self.layer_numbers = np.arange(len(self.layer_names))

    self.alpha = {'alluvium': alpha_a}
    self.k = {'alluvium': k_a}
    self.layer_tops = [z_sed]
    self.layer_names = ['alluvium_0']
    self.layer_lithologies = ['alluvium']
    self.layer_numbers = np.arange(len(self.layer_names))
    """

  def updateFluvialTopo_z(self):
    """
    Vertical incision or aggradation
    """
    self.z_ch_old = self.topo[-1,-1]
    self.dz = self.z_ch - self.z_ch_old
    if self.dz > 0:
      if self.z_ch >= 0:
        sys.exit("Full valley filling not supported")
      else:
        self.aggrade()
    elif self.dz < 0:
      self.incise()
    elif self.dz == 0:
      pass
    else:
      sys.exit("Warning: dz is not finite")
    #print self.topo
    #for layer_top in self.layer_tops:
      #print layer_top
    #print ""

  def updateFluvialTopo_y(self, constant=5):
    """
    Lateral migration
    """
    pass
    self.lateralErosionConstant(constant)
    
  def lateralErosionConstant(self, constant):
    """
    Constant lateral erosion with time 
    """
    # Start with a constant rate with time
    leftmost_at_channel_level = \
        np.nanmin( self.topo[:,0][np.round(self.topo[:,1], 5) == \
                                           np.round(self.z_ch, 5)] )
    #valley_middle_point = np.array([0, self.z_ch])
    old_valley_floor_edge = np.squeeze( self.topo[self.topo[:,0] ==
                                        leftmost_at_channel_level] )
    point = old_valley_floor_edge - np.array([constant, 0])
    #point = self.topo[-1] - np.array([constant, 0])
    print 'POINT', point
    #self.topo_updates = [valley_middle_point, old_valley_floor_edge, point]
    self.topo_updates = [point]
    
    # Figure out which layer needs to have an extra point added for the 
    # old x-position of the cliff... and add that point
    from_point = np.array([0, self.z_ch])
    inLayer = self.insideOrEnteringWhichLayer(point, from_point)
    inLayer_i = (self.layer_numbers == inLayer).nonzero()[0][0]
    self.layer_tops[inLayer_i] = \
          np.vstack(( self.layer_tops[inLayer_i], old_valley_floor_edge ))
    self.layer_tops[inLayer_i] = self.layer_tops[inLayer_i] \
                           [self.layer_tops[inLayer_i][:,0].argsort()]
    
    self.erode(point, old_valley_floor_edge)
    # Ensure that the sediments deposit appropriately, too.
    #alluv_layer_number = self.insideOrEnteringWhichLayer(point)
    #alluv_layer_tops = self.layer_tops[alluv_layer_number]
    #alluv_layer_tops[
    
  def layer_boundaries_noinf(self, layers, infinity_to_left=None, \
                             infinity_to_bottom=None, longreturn=False):
    points = np.vstack(layers)
    if infinity_to_left is None:
      minx_not_inf = np.min(points[np.isinf(points[:,0]) == False][:,0])
      infinity_to_left = minx_not_inf*1.2 - 1
    if infinity_to_bottom is None:
      miny_not_inf = np.min(points[np.isinf(points[:,1]) == False][:,1])
      infinity_to_bottom = miny_not_inf*1.2 - 1
    i=0
    for layer in layers:
      layer[:,0][np.isinf(layer[:,0])] = infinity_to_left
      layer[:,1][np.isinf(layer[:,1])] = infinity_to_bottom
    if longreturn:
      output = [layers, infinity_to_left, infinity_to_bottom]
    else:
      output = layers
    return output

  def layer_area_diff(self):
    """
    Assuming that the future layers (layer 2) will require equally or more
    extensive boundaries than the past layers (layer 1)
    """
    layers1 = copy.deepcopy(self.layer_boundaries_before)
    layers2 = copy.deepcopy(self.layer_boundaries)
    areas1 = []
    areas2 = []
    # Get boundaries and values to turn -np.inf into for lbni2
    lbni_out_2 = self.layer_boundaries_noinf(layers2, longreturn=True)
    lbni2 = lbni_out_2[0]
    lbni1 = self.layer_boundaries_noinf(layers1, 
                                        infinity_to_left = lbni_out_2[1],
                                        infinity_to_bottom = lbni_out_2[2])
    # Now get areas
    areas1 = self.get_all_areas(lbni1)
    areas2 = self.get_all_areas(lbni2)
    
    # Now compare them
    diff = []
    layer_numbers_both_times = np.hstack (( self.layer_numbers, 
                                            self.layer_numbers_before ))
    layer_numbers_both_times = np.unique(layer_numbers_both_times)
    for n in layer_numbers_both_times:
      # If these layers don't exist, will be comparing arbitrary boundaries...
      # For aggradation, probably not used, new layer is confined by valley.
      # For incision, new layers would just not include old layer, and can't
      #   erase anything that goes to infinity. So also just confined layers.
      #   SO OK.
      if (self.layer_numbers == n).any() and \
                              (self.layer_numbers_before == n).any():
        diff.append( areas2[self.layer_numbers == n] - \
                     areas1[self.layer_numbers_before == n] )
      elif (self.layer_numbers_before == n).any():
        diff.append( - areas1[self.layer_numbers_before == n] )
      elif (self.layer_numbers == n).any():
        # Shouldn't happen -- shouldn't run this while aggrading.
        # (Though possible in lateral migration phase of aggradation)
        diff.append( areas2[self.layer_numbers == n] )
      else:
        sys.exit()

    diff = np.squeeze(np.array(diff))
    return diff

  def diff_bedrock(self):
    darea = self.layer_area_diff()
    # Should only incise and lose layers for erosion
    layer_lithologies = np.array(self.layer_lithologies_before)
    isbedrock = (layer_lithologies != 'colluvium') * \
                (layer_lithologies != 'alluvium')
    # Layers should stay in order, so:
    darea_br = np.sum(darea[isbedrock])
    return darea_br

  def collluvial_pile(self):
    """
    Turn eroded material into a colluvial pile with an appropriate angle
    of repose.
    
    If the river is incising, assume that it removes the colluvium.
    Later, add this as an impediment to incision.
    
    If the river is widening, then pile colluvium to either:
    (a) an angle-of-repose pile equal to the volume of the deposit 
        plus a "fluffing" factor related to porosity
        (hard-coded to be equal to 0.35)
    (b) a pile that goes directly from the cliff top to the river
    
    Only do this for bedrock: alluvium is assumed to be removed completely
    (though these same area calculations will be important to determine how
     much of the alluvial bank the river can move through in a given time)
    """
    # Constants
    lambda_p = 0.35
    angleOfRepose = self.alpha['colluvium']
    m = - np.tan( (np.pi/180.) * angleOfRepose)
    # Volume eroded
    darea_br = self.diff_bedrock()
    A_deposit = -1/(1-lambda_p) * darea_br
    # Cliff top = first lip above river level.
    cliff_top_point = self.topo[self.topo[:,1] > self.z_ch][-1]
    cliff_bottom_point = self.topo[self.topo[:,1] == self.z_ch][0]
    h_cliff = cliff_top_point[1] - cliff_bottom_point[1]
    dy_cliff_base_to_channel = - cliff_bottom_point[0]
    # Will it fit?
    alpha = angleOfRepose * np.pi / 180.
    beta = np.pi - np.arctan2( cliff_top_point[1] - cliff_bottom_point[1], \
                               cliff_top_point[0] - cliff_bottom_point[0] )
                           
    #h = ( 2 * A_deposit * np.tan(alpha) * np.sin(beta) / np.cos(np.pi/2. - beta ) )**.5
    #h = ( 2 * A_deposit * np.tan(alpha))**.5 <-- doesn't take br slope into account
    height = ( 2*A_deposit * np.tan(alpha) / (1 - np.tan(alpha) * (1/np.tan(beta))) )**.5
    base = 2 * A_deposit / height
    enough_space = (height <= h_cliff) * (base <= dy_cliff_base_to_channel)
    if enough_space:
      # Intercept with fluvial layer
      #b = point[1] - m*point[0]
      #self.findIntersections(m, b, topo)
      z_at_deposit_top = self.z_ch + height
      x_at_deposit_top = self.piecewiseLinearAtZ(z_at_deposit_top, self.topo)
      z_at_deposit_bottom = self.z_ch
      x_at_deposit_bottom = base - dy_cliff_base_to_channel
    # IN FUTURE, MUST FIND A WAY TO ACCOUNT FOR THIS MATERIAL!
    elif (base > dy_cliff_base_to_channel):
      x_at_deposit_bottom = 0
      z_at_deposit_bottom = self.z_ch
      #self.findIntersections(m, b, self.topo)
      x_at_deposit_top, z_at_deposit_top = \
          self.findLikelyBestIntersection( m=m, b=self.z_ch,
          piecewiseLinear=self.topo,
          starting_point=np.array([x_at_deposit_bottom, z_at_deposit_bottom]) )
    elif (height > h_cliff):
      x_at_deposit_top, z_at_deposit_top = cliff_top_point
      x_at_deposit_bottom, z_at_deposit_bottom = \
        self.findIntersections(m, 
                               b = z_at_deposit_top - m*x_at_deposit_top,
                                  piecewiseLinear = self.topo)[-1]

    colluvial_layer_top = np.array([[x_at_deposit_top, z_at_deposit_top],
                                    [x_at_deposit_bottom, z_at_deposit_bottom]])
    
    
    
      
    """
    # Triangle plus rectangle.
    rock_triangle_base_width = cliff_bottom_point[0] - cliff_top_point[0]
    full_base_width = - cliff_top_point[0]
    height = cliff_top_point[1] - self.z_ch
    max_area = 0.5 * (full_base_width - rock_triangle_base_width) * height
    # max_area is always 0 for incision through something with a steeper
    # angle of repose.

    if max_area == 0:
      print "Did you just incise? No space to depsit colluvium -- or too steep."
      colluvial_layer_top = None
    elif max_area <= A_deposit:
      colluvial_layer_top = np.vstack((cliff_top_point,
                                       np.array([0, self.z_ch]) ))
    elif max_area > A_deposit:
      # Angle diagram
      alpha = angleOfRepose * np.pi / 180.
      beta = np.pi - np.arctan2( cliff_top_point[1] - cliff_bottom_point[1], \
                                 cliff_top_point[0] - cliff_bottom_point[0] )
      h = ( 2 * A_deposit * np.sin(beta) / np.cos(np.pi/2. - beta ) )**.5

      # commented earlier
      # Rearrange triangle equation
      # b_full = 2 * (A/h) + b_bedrock_triangle
      deposit_width = 2 * (A_deposit / height) + rock_triangle_base_width
      x_colluv_end = cliff_top_point[0] + deposit_width
      colluvial_layer_top = np.vstack((cliff_top_point,
                                       np.array([x_colluv_end, self.z_ch]) ))
    """

    if colluvial_layer_top is not None: # unnecessary if-statement, now
      # Update layers
      self.layer_tops.append(colluvial_layer_top)
      self.layer_numbers = np.arange(len(self.layer_numbers)+1)
      self.layer_lithologies.append('colluvium')
      number_of_layers_of_colluvium = len(fnmatch.filter(self.layer_names, 'colluvium*'))
      new_layer_name = 'colluvium_'+str(number_of_layers_of_colluvium)
      self.layer_names.append(new_layer_name)
      # Update topo
      topo = self.topo.copy()
      i = 0
      rows_to_delete = []
      for point in topo:
        topo_at_colluv = self.piecewiseLinearAtX(point[0], colluvial_layer_top)
        if np.round(topo_at_colluv, 5) >= np.round(point[1], 5):
          print 'pop'
          rows_to_delete.append(i)
        i += 1
      topo = np.delete(topo, rows_to_delete, axis=0)
      topo = np.vstack((topo, colluvial_layer_top))
      topo = topo[ topo[:,0].argsort()]
      self.topo = topo.copy()
      self.layer_boundaries = self.calc_layer_boundaries()
      self.calc_layer_bottoms()

  def get_all_areas(self, layers):
    areas = []
    lbni = self.layer_boundaries_noinf(layers)
    for layer in lbni:
      areas.append( self.area(layer) )
    areas = np.array(areas)
    return areas

  def area(self, layer):
    cp = 0
    x = layer[:,0]
    y = layer[:,1]
    for i in range(len(layer)-1):
      # Cross product
      cp += x[i]*y[i+1] - x[i+1]*y[i]
    area = 0.5 * np.abs(np.sum(cp))
    return area

  def channelGeometry(self):
    pass

  def incise(self):
    """
    When river incises, compute erosion and angle-of-repose destruction of
    material above.
    When collapse of bedrock does happen, it creates an angle-of-repose pile
    of sediments in the real world. Represent this in some way?
    """
    point = np.array([0, self.z_ch])
    self.topo_updates = []
    self.erode(point)
    
  def erode(self, point, old_valley_floor_edge=None):
    """
    Erode layers -- good for incision or lateral motion
    """
    from_point = None # NULL in future
    # "topo_updates" holds new points that modify layers until the end,
    # when we are ready to update the whole system all at once
    chosen_layer_numbers = []
    #chosen_layer_numbers.append(self.insideWhichLayer(point))
    topodefflag = False
    # Old layers
    #self.topo_before = self.topo.copy()
    #self.layer_tops_before = self.layer_tops[:]
    #self.layer_bottoms_before = self.layer_bottoms[:]
    self.layer_boundaries_before = self.layer_boundaries[:]
    self.layer_numbers_before = self.layer_numbers[:]
    self.layer_names_before = self.layer_names[:]
    self.layer_lithologies_before = self.layer_lithologies[:]
    #
    ii = 0
    while point is not None:
      #if (point == -np.inf).any():
      #  break
      #print "&&&&", point, from_point
      point = np.squeeze(point)
      print point, from_point
      inLayer = self.insideOrEnteringWhichLayer(point, from_point)
      from_point = point.copy()
      if inLayer is None:
        # An option: is above everything
        z_topo = self.piecewiseLinearAtX(point[0], self.topo)
        if point[1] > z_topo:
          #print "OUT!!!"
          # Above all? Then should be at the top.
          # After going above topographic surface, nothing to do.
          point = None # break out of loop.
        else:
          # Otherwise, just eroded exactly through a layer.
          # Look at next layer below
          z_topo = []
          for layer_top in self.layer_tops:
            z_topo.append(self.piecewiseLinearAtX(point[0], layer_top))
          z_topo = np.array(z_topo)
          # And pick upslope layer, if in doubt.
          inLayer = self.layer_numbers[ \
                         np.round(z_topo, 5) == np.round(point[1], 5)][0]
      
      elif inLayer == -1:
        # Then, it is entering the outside world; use topography to find a
        # temporary "angle of repose".
        x_left = self.topo[:,0][self.topo[:,0] < point[0]]
        # geq for right b/c leftmost point must be away from channel.
        # Was going to calculate angle, but really no need. Just find next
        # point to the right that is on topo
        #x_right = self.topo[:,0][self.topo[:,0] >= point[0]]
        #topo_left = self.topo[ np.round(self.topo[:,0], 5) == \
        #                       np.round(np.max(x_left), 5) ]
        #topo_right = self.topo[ np.round(self.topo[:,0], 5) == \
        #                        np.round(np.min(x_right), 5) ]
        #point = self.topo[ np.round(self.topo[:,0], 5) == \
        #                       np.round(np.max(x_left), 5) ]
        #continue
        topo_left = self.topo[ np.round(self.topo[:,0], 6) <= \
                               np.round(np.max(x_left), 6) ]
        for item in topo_left[::-1]:
          self.topo_updates.append(item)
        break
      
      """
      if type(inLayer) is list:
        # Then, it is entering the outside world; use topography to find a
        # temporary "angle of repose".
        x_left = self.topo[:,0][self.topo[:,0] < point[0]]
        # geq for right b/c leftmost point must be away from channel.
        # Was going to calculate angle, but really no need. Just find next
        # point to the right that is on topo
        #x_right = self.topo[:,0][self.topo[:,0] >= point[0]]
        #topo_left = self.topo[ np.round(self.topo[:,0], 5) == \
        #                       np.round(np.max(x_left), 5) ]
        #topo_right = self.topo[ np.round(self.topo[:,0], 5) == \
        #                        np.round(np.min(x_right), 5) ]
        point = self.topo[ np.round(self.topo[:,0], 5) == \
                               np.round(np.max(x_left), 5) ]
      """
      
      if point is not None:
        # Carry on, cowboy/girl!
        # slope-intercept
        angleOfRepose = self.alpha[self.layer_lithologies[inLayer]]
        m = - np.tan( (np.pi/180.) * angleOfRepose)
        #print m
        b = point[1] - m*point[0]
        # Find intersection with each layer
        intersections = []
        for i in self.layer_numbers:
          # list of 1D numpy arrays
          intersections.append(self.findLikelyBestIntersection(m=m, b=b, \
                                    piecewiseLinear=self.layer_tops[i], \
                                    starting_point=point))
        intersections = np.squeeze(np.array(intersections))
        # Don't want to round the thing that goes into everything else!
        # Keep this high precision and round later.
        intersections = np.round(intersections, 6)
        # First, use only those points are above the point in question
        # now handled in function
        #intersections[intersections[:,1] <= point[1]] = np.nan
        # if nothing above, then we are at the top
        if np.isnan(intersections).all():
          # Break out of loop
          point = None
        else:
          path_lengths = ( (intersections[:,0] - point[0])**2 \
                         + (intersections[:,1] - point[1])**2 )
          path_lengths_nonzero = path_lengths[path_lengths > 0]
          # And of these nonzero paths, find the shortest that is not at the 
          # same location as the point, and this is the
          # chosen intersection
          # WILL THIS CAUSE PROBLEMS WHEN WE INCISE EXACTLY TO THE LEVEL OF A 
          # LAYER? PROBABLY NOT -- JUST LOOKING ABOVE IT.
          # Chosen layer number will work here because self.layer_numbers is
          # ordered just by a np.arange (so same)
          chosen_layer_number = (path_lengths == \
                                 np.nanmin(path_lengths_nonzero)).nonzero()[0][0]
          chosenIntersection = intersections[chosen_layer_number]
          chosen_layer_numbers.append(chosen_layer_number)
          self.topo_updates.append(chosenIntersection)
          # For lateral erosion:
          # Lateral beveled strath alongside alluvial valley
          # First time through "while" loop --> first point added to topo.
          # Update the layers to include a point at the base of the new slope.
          # Think just first time so it's only done once -- so perhaps move it 
          # elsewhere?
          # Yes, up in the channel widening main code.
          """
          if (ii == 0) and (old_valley_floor_edge is not None):
            iii = (self.layer_numbers == chosen_layer_number).nonzero()[0][0]
            self.layer_tops[iii] = \
              np.vstack(( self.layer_tops[iii], old_valley_floor_edge ))
            self.layer_tops[iii] = self.layer_tops[iii] \
                                   [self.layer_tops[iii][:,0].argsort()]

          """
          # Now note that chosenIntersection is the new starting point
          point = chosenIntersection.copy()
          ii += 1
          
    if topodefflag is False:
      # Wait until the end to update the cross-sectional profile

      #print "TOPO UPDATES"
      #print self.topo_updates
      self.topo = self.newIncisedTopo(self.topo_updates)
      
      # Completely remove all layers that lie entirely above topography
      self.layer_boundaries = self.calc_layer_boundaries() # refresh      
      for n in range(len(self.layer_boundaries)):
        layer = self.layer_boundaries[n]
        topo_at_layer_points = []
        for point in layer:
          topo_at_layer_points.append(self.piecewiseLinearAtX(point[0], self.topo))
        topo_at_layer_points = np.array(topo_at_layer_points)
        if (topo_at_layer_points < layer[:,1]).all():
          #print n
          self.layer_numbers = np.arange(len(self.layer_numbers)-1) # renumber
                                   # to keep indices and layer_numbers the same
          self.layer_names.pop(n)
          self.layer_lithologies.pop(n)
          self.layer_tops.pop(n)

      # Vertices from topo must be added to layer tops
      # But ONLY those that lie above the layer bottoms.
      # -- this was source of earlier problem --
      # -- (channel added to upper layers) --
      for i in range(len(self.layer_tops)):
        layer_top = self.layer_tops[i]
        for topo_point in self.topo:
          x_topo = topo_point[0]
          y_topo = topo_point[1]
          y_layer_top = self.piecewiseLinearAtX(x_topo, layer_top)
          # Point must be inside layer: below top
          if (y_topo <= y_layer_top):
            # And be above its original bottom, if applicable
            if i < len(self.initial_layer_bottoms):
              y_layer_bottom = self.piecewiseLinearAtX(x_topo, \
                                             self.initial_layer_bottoms[i])
              if y_topo >= y_layer_bottom:
                #print topo_point
                self.layer_tops[i] = np.vstack((self.layer_tops[i], topo_point))
            # Unless this was not an original layer.
            else:
              self.layer_tops[i] = np.vstack((self.layer_tops[i], topo_point))
      
      # Then, remove all layer top points that lie above topo
      for i in range(len(self.layer_tops)):
        layer_top_trimmed = []
        for point in self.layer_tops[i]:
          x_layer_top = point[0]
          y_layer_top = point[1]
          #print "***"
          #print x_layer_top, self.topo, "***"
          y_topo = self.piecewiseLinearAtX(x_layer_top, self.topo)
          if y_topo >= y_layer_top:
            #print point
            layer_top_trimmed.append(point)
        #print np.array(layer_top_trimmed)
        self.layer_tops[i] = np.array(layer_top_trimmed)

      # Then update topo to agree with layer tops;
      # Should really do this in newIncisedTopo or somewhere.
      

      # Must check where channel incises through multiple layers, and ensure
      # that no layer is incised beyond its bottom
      # (I do this below: ensure that it is not too deep)
      
      """
      # NECESSARY CRUFT IN NEXT SECTION

      # Find bottoms of geologic layers at x=0
      # This is to ensure we don't incise past these
      layer_tops_at_x0 = []
      for j in range(len(self.layer_tops)):
        layer_top = self.layer_tops[j]
        layer_tops_at_x0.append( self.piecewiseLinearAtX(0, layer_top) )
      layer_tops_at_x0 = np.array(layer_tops_at_x0)
      _y_layer_bottom = []
      for i in range(len(layer_tops_at_x0)):
        try:
          _y_layer_bottom.append(np.max(layer_tops_at_x0[ \
                                 layer_tops_at_x0 < layer_tops_at_x0[i]]))
        except:
          _y_layer_bottom.append(-np.inf) # change to nan? sometimes layer just doesn't exist here anymore!
      #y_at_layers = np.array(y_at_layer)



      # Layers must be at the minimum elevation at each point
      # Minimum calculated between:
      #    - Layer elevation at point
      #    - Topography at point
      #    - Elevation of next layer below at point
      for i in range(len(self.layer_tops)):
        layer_top = self.layer_tops[i]
        final_points = []
        _x_values = list(set(layer_top[:,0]))
        for _x in _x_values:
          _y_in_layer = layer_top[:,1][layer_top[:,0] == _x]
          _y = np.min(layer_top[:,1][layer_top[:,0] == _x])
          # Ensure that it is not too deep
          if _x == 0:
            _y = np.max((_y, _y_layer_bottom[i]))
          final_points.append([_x,_y])
        layer_top = np.array(final_points)
        self.layer_tops[i] = layer_top
      """
      
      # Remove duplicate points
      self.layer_tops = self.rmdup_rows(self.layer_tops)
      self.topo = self.rmdup_rows(self.topo)
      
      # Then sort it all
      for i in range(len(self.layer_tops)):
        layer_top = self.layer_tops[i]
        self.layer_tops[i] = layer_top[layer_top[:,0].argsort()]

      # NOT SURE IF THIS IS NEEDED!
      # Then check for unnecesssary points and remove them
      for i in range(len(self.layer_tops)):
        layer_top = self.layer_tops[i]
        slopes = np.diff(layer_top[:,1]) / np.diff(layer_top[:,0])
        slopes = np.round(slopes, 5)
        # Those with constant slopes on both sides are not needed
        not_needed = np.hstack(( False, slopes[:-1] == slopes[1:], False ))
        self.layer_tops[i] = self.layer_tops[i][not_needed == False]
      

  def rmdup_rows(self, layers):
    """
    removing duplicates following the answer at:
    http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array/
    """
    # Single layer or multiple layers in list or other structure?
    output = []
    if type(layers) is np.ndarray:
      layers = [layers]
      _wasarray = True
    else:
      _wasarray = False
    for layer in layers:
      layer = self.unique_rows(layer)
      output.append(layer)
    if _wasarray:
      output = output[0]
    return output
    
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
    
    ############################
    # Horizontal alluvial fill #
    ############################
    x_valley_wall = self.piecewiseLinearAtZ(self.z_ch, self.topo)
    aggraded_surface = np.array([[x_valley_wall, self.z_ch],
                                 [0, self.z_ch]])
    oldtopo = self.topo[self.topo[:,-1] > self.z_ch]
    topo = np.vstack((oldtopo, aggraded_surface))
    self.topo = topo.copy()

    # Add this to the list of layers
    self.layer_tops.append(aggraded_surface)
    self.layer_numbers = np.arange(len(self.layer_numbers)+1)
    self.layer_lithologies.append('alluvium')
    number_of_layers_of_alluvium = len(fnmatch.filter(self.layer_names, 'alluvium*'))
    new_layer_name = 'alluvium_'+str(number_of_layers_of_alluvium)
    self.layer_names.append(new_layer_name)
    
    #######################################
    # Combine adjacent layers of alluvium #
    #######################################
    touching_layers = self.layersTouchedByLayer(new_layer_name)
    #print ""
    #print touching_layers
    #if touching_layers == self.layers_touching.any()
    layers_touched_logical = np.in1d(self.layer_numbers, touching_layers)
    layer_lithologies_touched = np.array(self.layer_lithologies) \
                                                        [layers_touched_logical]
    alluvial_layer_numbers_touched = np.array(touching_layers)[ \
                                    layer_lithologies_touched == 'alluvium']
    new_layer_number = self.layer_numbers[-1]
    
    touching_alluvial_layer_numbers = np.hstack((alluvial_layer_numbers_touched,\
                                                new_layer_number))
    # Pick all top points
    points_combined = []
    for i in touching_alluvial_layer_numbers:
      points_combined.append(self.layer_tops[i])
    points_combined = np.vstack(points_combined)
    combined_layer_top = []
    for point in points_combined:
      point_geq_layer = []
      for n in touching_alluvial_layer_numbers:
        # Append on two conditions:
        # 1. > all others
        # 2. nan (i.e. no other in this place
        gt_cond = np.round(point[1], 5) >= np.round( self.piecewiseLinearAtX(point[0], 
                                                      self.layer_tops[n]), 5)
        nan_cond = np.isnan( self.piecewiseLinearAtX(point[0], 
                                                      self.layer_tops[n]) )
        if gt_cond or nan_cond:
          point_geq_layer.append(True)
        else:
          point_geq_layer.append(False)
      if np.array(point_geq_layer).all():
        #print point
        combined_layer_top.append(point)
    combined_layer_top = np.array(combined_layer_top)
    combined_layer_top = self.rmdup_rows(combined_layer_top)
        
    # Remove layers with higher layer numbers
    for n in touching_alluvial_layer_numbers[touching_alluvial_layer_numbers!=
                                     np.min(touching_alluvial_layer_numbers)]:
      print n
      lni = int((self.layer_numbers == n).nonzero()[0])
      self.layer_numbers = np.delete(self.layer_numbers, lni)
      self.layer_names.pop(lni)
      self.layer_lithologies.pop(lni)
      self.layer_tops.pop(lni)
    self.layer_tops[np.min(touching_alluvial_layer_numbers)] = \
                                                           combined_layer_top

    # Update list of layer boundaries after aggradation
    self.layer_boundaries = self.calc_layer_boundaries()
    
    """
    # Will break now that layers are updated earlier on -- will see an
    # extra layer
    
    ######################################################
    # Check if we are adjacent to another alluvial layer #
    ######################################################
    
    # 1. Mark alluvial layers
    layer_is_alluvium = np.array(self.layer_lithologies) == 'alluvium'
    # Need to specify axis -- DEPRECATION WARNING
    alluv_layers = list(np.array(self.layer_tops)[layer_is_alluvium])
    alluv_layer_numbers = self.layer_numbers[layer_is_alluvium]
    
    # 2. Find number of adjacent layer
    contacts_layer_number = self.insideOrEnteringWhichLayer \
                                                 (aggraded_surface[0], \
                                                  aggraded_surface[1])

    # 3. Do a check that alluvium touches a layer
    if contacts_layer_number is None:
      sys.exit("Alluvium spilling out of valley?")

    # 4. Find if adjacent layer is alluvium
    
    
    #    If it is, append it.
    if self.layer_lithologies[contacts_layer_number] == 'alluvium':
      tmplayer = self.layer_tops[contacts_layer_number][:]
      tmplayer = tmplayer[tmplayer[:,1] > aggraded_surface[0,1]]
      tmplayer = np.vstack((tmplayer, aggraded_surface))
      tmplayer = tmplayer[tmplayer[:,-1] >= self.z_ch] # take only points
                                                       # not buried under alluv
      self.layer_tops[contacts_layer_number] = tmplayer[:]
    #    If it is not, make a new layer out of it
    else:
      self.layer_tops.append(aggraded_surface)
      self.layer_numbers = np.arange(len(self.layer_numbers)+1)
      self.layer_lithologies.append('alluvium')
      number_of_layers_of_alluvium = len(fnmatch.filter(self.layer_names, 'alluvium*'))
      self.layer_names.append('alluvium_'+str(number_of_layers_of_alluvium))
    # 5. (For the future): Find if you have re-connected two separated layers
    #    of alluvium. If you have, connect them. Should write a check here to
    #    use self.layers to see if any points are shared.
    #    OR even simplify this whole thing by always making that check!
    #    Points shared -- any parts of borders touching!
    """
  
  def layersTouchedByLayer(self, layer_name_or_number):
    if type(layer_name_or_number) is int:
      n = layer_name_or_number
    elif type(layer_name_or_number) is str:
      n = self.layer_numbers[np.array(self.layer_names) == layer_name_or_number]
      n = int(n)
    else:
      sys.exit("Integer or string required.")
    self.layer_boundaries= self.calc_layer_boundaries() # refresh
    layers_touching = []
    # NOTE -- WILL HAVE TO UPDATE THIS TO CYCLE THROUGH ALL LAYERS IF I HAVE
    # REALLY STRANGE GEOMETRIES -- NOT CURRENTLY EXPECTED.
    for point in self.layer_boundaries[n]:
      x = point[0]
      y0 = point[1]
      for i in range(len(self.layer_tops)):
        layer_top = self.layer_tops[i]
        if i != n: # not the same layer
          y1 = self.piecewiseLinearAtX(x, layer_top)
          if y1 == y0:
            layers_touching.append(i)
    layers_touching = list(set(layers_touching))
      
    return layers_touching

  def newAggradedTopo(self):
    topo = []
    topo.append([0, self.z_ch])
    
    for point in layers:
      if point[-1] >= self.z_ch:
        topo.append(list(point))
    # Get our -infinity
    # OLD: topo.append(list(self.layer_tops[-1][0]))
    # but valley fill comes after this.
    topo.append(self.topoBCinf())
    topo = np.array(topo)[::-1]
    topo = self.rmdup_rows(topo)
    topo = topo[ topo[:,0].argsort()]
    # Not final topo -- intermediate step.
    # So don't update self.topo
    return topo
  
  def topographicProfile(self, layers):
    # Topographic profile
    # Pick only the highest points at each position
    topo = []
    topoPoints = []
    for layer in layers:
      layerPoints = []
      for point in layer:
        layer_elevations_at_x = []
        for layer in layers:
          layer_elevations_at_x.append( self.piecewiseLinearAtX(point[0], layer) )
        layer_elevations_at_x = np.array(layer_elevations_at_x)
        layer_elevations_at_x = \
              layer_elevations_at_x[np.isnan(layer_elevations_at_x) == False]
        if (np.round(point[1], 5) >= np.round(np.array(layer_elevations_at_x), 5)).all():
          if (point[0] == 0) and (point[1] >= self.z_ch):
            pass
          else:
            layerPoints.append(point)
      if len(layerPoints) > 0:
        topoPoints.append(np.array(layerPoints))
    topoPoints.append(np.array([[0, self.z_ch]]))
    topoPoints = np.vstack(topoPoints)
    # reverse
    topoPoints = topoPoints[::-1]
    # Backwards (i.e. topmost first); pick first point seen
    for i in range(len(topoPoints)):
      layerPoints = topoPoints[i]
      if layerPoints.ndim == 1:
        layerPoints = np.expand_dims(layerPoints, 0)
      for point in layerPoints:
        try:
          inlist = np.sum(np.product(np.asarray(topo) == point, axis=1))
        except:
          inlist = np.sum(np.product(np.asarray(topo) == point, axis=0))
        if inlist:
          pass
        else:
          topo.append(point)
    topo = np.array(topo)[::-1]
    self.topo = topo
    
  def newIncisedTopo(self, layers):
    """
    Intermediate step -- compute new topo based on given incision.
    This is not the final topography -- but should be close.
    """

    #self.topo[:-1][self.topo[:-1,0] > np.max(layers[:,0])], \
                                              
    # 1. Add new channel
    layers = np.array(layers)
    #print 'layers\n',layers
    #print '***', np.max(layers[:,0]), np.min(layers[:,0])
    topo = np.vstack(( self.topo[self.topo[:,0] < np.min(layers[:,0])], \
                       np.array([[0, self.z_ch]]) ))
    
    # 2. Add other computed points (intersections)
    topotmp = self.topo.copy()
    for point in layers:
      y_topo_at_point = self.piecewiseLinearAtX(point[0], topotmp)
      #if point[1] <= y_topo_at_point:
      topo = np.vstack(( topo, point)) # probably slow
    
    # 3. Place in order    
    topo = topo[topo[:,0].argsort()]
    #print '***', topo

    # 4. Then check for unnecesssary points and remove them
    slopes = np.diff(topo[:,1]) / np.diff(topo[:,0])
    # Those with constant slopes on both sides are not needed
    not_needed = np.hstack(( False, slopes[:-1] == slopes[1:], False ))
    topo = topo[not_needed == False]
      
    return topo

    """
    topo = self.topo
    topo = []
    topo.append([0, self.z_ch])
    for point in layers:
      if point[-1] >= self.z_ch:
        topo.append(list(point))
    # Get our -infinity
    # OLD: topo.append(list(self.layer_tops[-1][0]))
    # but valley fill comes after this.
    topo.append(self.topoBCinf())
    topo = np.array(topo)[::-1]
    topo = self.rmdup_rows(topo)
    topo = topo[ topo[:,0].argsort()]
    # Not final topo -- intermediate step.
    # So don't update self.topo
    return topo
    """
    
  def topoBCinf(self):
    """
    Provides the topography of the surface far from the model domain
    """
    points = np.vstack(self.layer_tops)
    left = points[np.isinf(points[:,0])]
    out = left[left[:,1] == np.max(left[:,1])]
    out_ls = list(np.squeeze(out))
    return out_ls

  def topoPlot(self, linestyle='-'):
    # Should make ylim 0 at start to do this properly
    topoFinite = self.topo.copy()
    topoFinite[:,1] *= -1
    if np.isinf(topoFinite).any():
      if len(topoFinite) > 2:
        xmin = self.topo[1,0]
      else:
        xmin = -1E3 # arbitrarily large
      topoFinite[0,0] = xmin * 1000 # arbitrarily large number
    else:
      xmin = self.topo[0,0]
    yrange = np.max(self.topo[:,1]) - np.min(self.topo[:,1])
    plt.plot(topoFinite[:,0], topoFinite[:,1], linestyle)
    # ylim
    proposed_ylim = np.array([-0.1*yrange + np.min(topoFinite[:,1]), \
                               0.1*yrange + np.max(topoFinite[:,1])])
    both_ylims = np.vstack(( proposed_ylim, np.array(plt.ylim()) ))
    ylim_top = np.max(both_ylims)
    ylim_bottom = np.min(both_ylims)
    plt.ylim(( ylim_bottom, ylim_top ))
    plt.xlim((1.2*xmin, 0))
    plt.xlabel('Cross-valley distance [m]', fontsize=16, fontweight='bold')
    plt.ylabel('Distance below plateau surface [m]', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()

  def layerPlot(self, twoSided=False):
    fig, ax = plt.subplots()
    layers = self.calc_layer_boundaries()
    """
    if twoSided:
      for i in range(len(layers)):
        #layers[i] = np.vstack(( self.layer_tops[i],
        #                        np.array([-1, 1]) * self.layer_tops[i][::-1],
        #                        np.array([-1, 1]) * self.layer_bottoms[i][::-1],
        #                        self.layer_bottoms[i] ))
        layers[i] = np.vstack(( layers[i],
                                np.array([-1, 1]) * layers[i][::-1] ))
    """
    layers = self.layer_boundaries_noinf(layers)
    i=0
    #color_cycle = ['r', 'g', 'b', 'c', 'm', 'y']
    color_cycle = ['rosybrown', 'lightgreen', 'lightskyblue', 'c', 'm', 'y']
    for layer in layers:
      layer[:,1] *= -1 # depth increases dowrnward
      shape = plt.Polygon(layer, facecolor=color_cycle[i%len(color_cycle)], 
                          edgecolor='k', linewidth=2, label=self.layer_names[i])
      #shape = plt.Polygon(layer, edgecolor='k', facecolor='w', label=self.layer_names[i])
      stratum = ax.add_patch(shape)
      if self.layer_lithologies[i] == 'bedrock':
        stratum.set_hatch('//')
      elif self.layer_lithologies[i] == 'alluvium':
        stratum.set_hatch('..')
      elif self.layer_lithologies[i] == 'colluvium':
        stratum.set_hatch('oo')
      i+=1
    # Topo
    topoFinite = self.calc_topoFinite()
    # plotting limits
    all_points = np.vstack(layers)
    plt.xlim(( np.min(all_points[:,0]), np.max(all_points[:,0]) ))
    plt.ylim(( np.min(all_points[:,1]), np.max(all_points[:,1]) ))
    #plt.axis('scaled')
    plt.legend(loc='lower left')
    #plt.show()
    #labels = self.layer_lithologies
    #legend = plt.legend(labels, loc=(0.9, .95), labelspacing=0.1)
    #plt.setp(legend.get_texts(), fontsize='small')
    plt.plot(topoFinite[:,0], topoFinite[:,1], 'k-', linewidth=3)
    plt.xlabel('Cross-valley distance [m]', fontsize=16, fontweight='bold')
    plt.ylabel('Distance below plateau surface [m]', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()

  def calc_topoFinite(self):
    topoFinite = self.topo.copy()
    topoFinite[:,1] *= -1
    if np.isinf(topoFinite).any():
      if len(topoFinite) > 2:
        xmin = self.topo[1,0]
      else:
        xmin = -1E3 # arbitrarily large
    else:
      xmin = self.topo[0,0]
    topoFinite[0,0] = xmin * 1000 # arbitrarily large number
    return topoFinite

  def calc_layer_boundaries(self):
    """
    For each line segment:
      find all highest points that are below it.
      append these to a big line.
    Attach this to the layer_top stuff, as the layer bottom
    And then there is a closed boundary, and I can write methods to test what
    touches this or is inside it.
    """
    layers = []
    for layer_top in self.layer_tops:
      layers.append(self.calc_layer_boundary(layer_top))

    return layers
  
  def calc_layer_boundary(self, layer_top):

    #self.layers = []
    #for layer_top in self.layer_tops:
    # Set-up    
    left = layer_top[0,0]
    right = layer_top[-1,0]

    is_below = []
    # Create a list of layers that are above (0) or below (1) layer_top
    for i in range(len(self.layer_tops)):
      other_layer_top = self.layer_tops[i]
      for point in other_layer_top:
        layer_top_at_point = self.piecewiseLinearAtX(point[0], layer_top)
        #print layer_top_at_point
        #print layer_top_at_point, point[1], layer_top_at_point > point[1]
        if layer_top_at_point > point[1]:
          print layer_top_at_point, point
          is_below.append(i)
    is_below = sorted(list(set(is_below))) # rmv sorted for speed?
    #is_below_array = np.in1d(self.layer_numbers, is_below)
    
    # After this, find for each point, which piecewise linear layer top
    # of these candidates is the highest. Whichever one is will be the layer
    # bottom of the next layer above.
    potential_bottom_point_layers = []
    for layer_number in is_below:
      potential_bottom_point_layers.append(self.layer_tops[layer_number])
    
    # Then check for each point that lies below the layer in question, whether
    # it is at the top.
    # If it is, add it to a list of bottom points
    bottom_points = []
    for i in range(len(potential_bottom_point_layers)):
      layer_i = potential_bottom_point_layers[i]
      other_layers = potential_bottom_point_layers[:i] + \
                     potential_bottom_point_layers[i+1:]
      points = layer_i[(layer_i[:,0] >= left) * (layer_i[:,0] <= right)]
      for point in points:
        other_layer_tops_at_point = []
        for other_layer in other_layers:
          other_layer_tops_at_point.append(np.round(
                           self.piecewiseLinearAtX(point[0], other_layer), 5))
        #if (np.round(point[1], 5) > \
        #           np.array(other_layer_tops_at_point)).all():
          # Although the first cut looked for layers below, not every layer is
          # completely below.
          # So check that all new points are below.
          #print point, self.piecewiseLinearAtX(point[0], layer_top)
        if np.round(point[1], 5) < \
                 np.round(self.piecewiseLinearAtX(point[0], layer_top), 5):
          bottom_points.append(point)
    # For bottom layer
    if len(bottom_points) == 0:
      bottom_points = np.array([[0, -np.inf], [-np.inf, -np.inf]])
    else:
      bottom_points = np.vstack(bottom_points)
      # Sort R to L (greatest to least)
      bottom_points = bottom_points[ bottom_points[:,0].argsort() ][::-1]
    
    layer_boundary = np.vstack((layer_top, bottom_points))
    #self.layers.append(layer_boundary)
    
    return layer_boundary
  
  def calc_layer_bottoms(self):
    """
    A wrapper for "layer bottom" to update all layer bottoms
    """
    self.layer_bottoms = []
    for i in range(len(self.layer_tops)):
      self.layer_bottoms.append(self.calc_layer_bottom(self.layer_boundaries[i],
                                                  self.layer_tops[i]))

    return self.layer_bottoms
  
  def calc_layer_bottom(self, layer_boundary, layer_top):
    """
    Gives all the points that are needed to define a layer's bottom
    """
    layer_bottom = []
    for point in layer_boundary:
      if point[1] < self.piecewiseLinearAtX(point[0], layer_top):
        layer_bottom.append(point)
    # If left edge missing b/c it is a piece of the top, add it
    bottom_at_top_left = self.piecewiseLinearAtX(layer_top[0][0], np.array(layer_bottom))
    if np.isnan(bottom_at_top_left):
      layer_bottom.insert(0, layer_top[0])
    # If we're missing the last piece of the bottom because it's in the top,
    # add it.
    bottom_at_top_right = self.piecewiseLinearAtX(layer_top[-1][0], \
                                                np.array(layer_bottom))
    if np.isnan(bottom_at_top_right):
      layer_bottom.append(layer_top[-1])

    layer_bottom = np.array(layer_bottom)
    layer_bottom = layer_bottom[ layer_bottom[:,0].argsort()]
    
    return layer_bottom
    
    # This start/end part could also be assisted by left/right; keeping this
    # here as a note in case something goes wrong in this function
    #left = np.min(layer_boundary[:,0])
    #right = np.max(layer_boundary[:,0])

  def highest_points_below(self, pwl):
    """
    This may be leaky for finding layers directly below --
    looks at highest points below, but does not look to
    see whether another layer lies between.
    WATCH OUT FOR A BUG FROM THIS ONE!
    Also currently gives all layers' highest points -- so multiple from one
    layer possible.
    """

    # Set-up    
    highest_points_below = []
    left = pwl[0,0]
    right = pwl[-1,0]
    
    # Points at tops of layers
    layer_top_points = np.vstack(self.layer_tops)
    layer_top_points = layer_top_points[(layer_top_points[:,0] >= left) *
                                        (layer_top_points[:,0] <= right)]

    # Points at tops of layers
    for point in layer_top_points:
      ztop = self.piecewiseLinearAtX(point[0], pwl)
      if point[1] < ztop:
        highest_points_below.append(point)
    highest_points_below = self.unique_rows(np.vstack(highest_points_below))
    
    # Take highest point if multiple layers offer one
    #x_values = list(set(highest_points_below[:,0]))
    
    
    # Sort
    highest_points_below = highest_points_below[highest_points_below \
                                                [:,0].argsort()]
    return highest_points_below
      
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
  
  def distance(a,b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**.5

  def isPointOnSegment(point, segment):
    """
    Is point on line segment?
    """
    return distance(segment[0], point) + distance(segment[1], point) \
       == distance(segment[0], segment[1])
  
  def findLikelyBestIntersection(self, m, b, piecewiseLinear, starting_point):
    """
    Finds the closest intersection that is >= the current point and that is not 
    the current point.
    
    I think that this is fully generalized, except for the note at the bottom,
    but am not feeling sure enough -- hence "Likely" best
    """
    intersections = self.findIntersections(m, b, piecewiseLinear)
    # First check if there is anything to work with
    if intersections.size == 0:
      intersection = np.array([np.nan, np.nan])
    # Then if there is, see if it is an independent point
    else:
      intersections_at_or_above_point = \
          intersections[intersections[:,1] >= starting_point[1]]
      # remove any points that duplicate the starting point
      intersections_at_or_above_point = intersections_at_or_above_point[ \
          np.prod(intersections_at_or_above_point == starting_point, axis=1) \
          == 0]
      if intersections_at_or_above_point.size == 2:
        # One point
        intersection = intersections_at_or_above_point
      elif intersections_at_or_above_point.size < 2:
        intersection = np.array([np.nan, np.nan])
      else:
        intersection = intersections_at_or_above_point[ \
                       intersections_at_or_above_point[:,1] == \
                       np.min(intersections_at_or_above_point[:,1])]
      """
      # always going to be lowest point, so don't need full distance
      distances = []
      for point in intersections_at_or_above_point
      But what if more than one at the same elevation -- is this possible?
      """
    return np.squeeze(np.array(intersection))
    
  def findIntersections(self, m, b, piecewiseLinear):
    """
    Find intersection between two lines.
    
    m, b for angle-of-repose slope (m) and intercept (b) coming up from river,
         or, if this is after the first river--valley wall intersection,
         somewhere on the slope above the river
    
    piecewiseLinear: A piecewise linear set of (x,y) positions for the next 
                     geologic unit above the river. This should take the form:
                     ((x1,y1), (x2,y2), (x3,y3), ..., (xn, yn))
    
    """
    intersections = []
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
      if (np.round(xint, 5) >= np.round(xy0[0], 5)) and \
        (np.round(xint, 5) <= np.round(xy1[0], 5)):
        # You're done!
        # y-value plugging into one equations -- let's use the line from the 
        # starting point that is producing the erosion
        yint = m*xint + b
        # Because there is some numerical error created by defining yint
        # with such an equation let's add a rounding term: round to nearest
        # 1E-9 m (micrometer) -- because totally insignificant in these systems
        # --> Removing this rounding because it is causing errors when compared
        # with un-rounded values.
        # Apply rounding later!
        #intersections.append(np.round(np.array([xint, yint]), 5))
        intersections.append(np.array([xint, yint]))
    # Always return 2D array
    intersections = np.array(intersections, ndmin=2)
    return intersections

  def findIntersectionSegment(self, segment, piecewiseLinear):
    """
    Find intersections between a line segment and a piecewise linear line.
    Returns None-value if the intersection is off the line
    
    segment: endpoints of the line segment
    
    piecewiseLinear: A piecewise linear set of (x,y) positions for the next 
                     geologic unit above the river. This should take the form:
                     ((x1,y1), (x2,y2), (x3,y3), ..., (xn, yn))
    
    """
    m = (segment[1,1] - segment[0,1]) / (segment[1,0]-segment[0,0])
    b = segment[0,1] - m*segment[0,0]
    
    intersection_no_bounds = self.findIntersections(m, b, piecewiseLinear)
    
    x_inb = intersection_no_bounds[0]
    
    if (x_inb <= np.max(segment[:,0])) and (x_inb >= np.min(segment[:,0])):
      intersection = intersection_no_bounds
    else:
      intersection = None
      
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
      z = np.array([np.nan])
    else:
      # First, define line segment of interest
      xmin_pwl = np.max( pwl[:,0][pwl[:,0] <= x] )
      xmax_pwl = np.min( pwl[:,0][pwl[:,0] >= x] )
      z_xmin_pwl = (pwl[:,1][pwl[:,0] == xmin_pwl])
      if len(z_xmin_pwl) > 1:
        if (z_xmin_pwl == z_xmin_pwl[0]).all():
          z_xmin_pwl = z_xmin_pwl[0]
        else:
          print x, pwl
          sys.exit(">1 possible point at x; not a function!")
      z_xmax_pwl = (pwl[:,1][pwl[:,0] == xmax_pwl]) # in case two have it
      if len(z_xmax_pwl) > 1:
        if (z_xmax_pwl == z_xmax_pwl[0]).all():
          z_xmax_pwl = z_xmax_pwl[0]
        else:
          sys.exit(">1 possible point at x; not a function!")
      if (xmin_pwl == xmax_pwl) or (z_xmin_pwl == z_xmax_pwl):
        # The latter prevents double infinities in z from producing a nan
        # in else
        z = z_xmin_pwl
      else:
        # z = mx + b
        m = (z_xmax_pwl - z_xmin_pwl)/(xmax_pwl - xmin_pwl)
        # using max to avoid -inf
        b = z_xmax_pwl - m*xmax_pwl
        z = m*x + b
    
    # At this point, must be just one value, but this would be the place to 
    # add a check for it if I'm worried.
    
    # Maybe an array was input, maybe not
    try:
      return z[0]
    except:
      return z
    
  def piecewiseLinearAtZ(self, z, pwl):
    """
    Evaluates a piecewise linear expression to solve for x at a given z.
    z:   the z-value
    pwl: a 2-column numpy array, ([x, z]), that must contain at least
         four entries (two points define a line)
    This will make the line be as long as possible
    """
    
    # First check if x(z) not defined for this line.
    if z < np.min(pwl[:,0]) or z > np.max(pwl[:,0]):
      x = np.nan
    else:
      # First, define line segment of interest
      zmin_pwl = np.max( pwl[:,1][pwl[:,1] <= z] )
      zmax_pwl = np.min( pwl[:,1][pwl[:,1] >= z] )
      # Using np.max and np.min here because not necessarily a function in 
      # (z, x) space (horizontal lines); max/min value matters for alluviation
      # IF ANYTHING HAS TO BE CHANGED, IT WILL BE HERE!!!!!!!!!!!!!!!!!!!!!!!!!S
      x_zmin_pwl = float(np.min(pwl[:,0][pwl[:,1] == zmin_pwl]))
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
    self.layer_tops[self.layer_lithologies == 'alluvium']
  
  #, layer_tops=None, layer_bottoms=None, layer_numbers=None
  def insideOrEnteringWhichLayer(self, point, from_point=None):
    """
    Point is (x,z)
    This script will return which layer the point is in.
    Because it is used to find which angle is the proper angle of repose
    for the rock/sediment/soil/etc. above, it will look where the last point
    was as well to see which layer it is entering.
    
    The importance of this script lies in the fact that each time the river 
    incises, the point will end up in the ground, and will have to project out 
    of that. In plausible cases, it might end up projecting back up through 
    multiple materials.
    
    Defaults to work on the standard list of layers.
    """
    
    #print 'point', point

    layer_tops = self.layer_tops
    self.calc_layer_bottoms()
    layer_bottoms = self.layer_bottoms
    layer_numbers=self.layer_numbers

    #    if layer_tops is None:
    #      layer_tops = self.layer_tops
    #    if layer_bottoms is None:
    #      self.calc_layer_bottoms()
    #      layer_bottoms = self.layer_bottoms
    #    if layer_numbers is None:
    #      layer_numbers=self.layer_numbers
    
    layer_number = None # Flag before being actual number

    if type(layer_tops) == np.ndarray:
      layer_tops = [layer_tops]
    if type(layer_bottoms) == np.ndarray:
      layer_bottoms = [layer_bottoms]
    
    layer_top_elevations_at_point = []
    layer_bottom_elevations_at_point = []
    for i in range(len(layer_tops)):
      #print '***', point
      layer_top_elevations_at_point.append(self.piecewiseLinearAtX(point[0], layer_tops[i]))
      layer_bottom_elevations_at_point.append(self.piecewiseLinearAtX(point[0], layer_bottoms[i]))
    layer_top_elevations_at_point = np.array(layer_top_elevations_at_point)
    layer_bottom_elevations_at_point = np.array(layer_bottom_elevations_at_point)
    # Need to check
    # 1. is point above all layers? if so, error.
    #        --> check that this works with aggradation
    # 2. is point the first one? If so, method of getting slopes (later)
    #        won't work, and we have to treat it differently.
    #    We know that this is always inside a layer, so this can help.
    #    In fact, use this as an internal test!
    
    # 1. Check if point is above all layers
    #    Later move this to the end, for efficiency.
    with np.errstate(invalid='ignore'):
      layers_above_point = layer_top_elevations_at_point > point[1]
      point_above_all_layers = (layer_top_elevations_at_point < point[1]).all()
    #print layer_top_elevations_at_point
    #print point[1]
    #print (layer_top_elevations_at_point != point[1]).all()
    layer_tops_at_point_elevation = (np.round(layer_top_elevations_at_point, 5) \
                                     == np.round(point[1], 5))
    layer_bottoms_at_point_elevation = (np.round(layer_bottom_elevations_at_point, 5) \
                                     == np.round(point[1], 5))
    layers_at_point_elevation = layer_tops_at_point_elevation + \
                                layer_bottoms_at_point_elevation
    if point_above_all_layers:
      # Must be above everything, the point is in free space!
      layer_elevation_point_is_inside = None
      layer_number = None
      print "Warning: POINT ABOVE ALL LAYERS!"
    # 2. This should also work if it is the first point.
    #    Check if this is the first point in the series
    elif (layers_at_point_elevation).any() == False:
      # Point inside a layer
      # Point does not lie on a layer top, but there should be at least
      # one layer above it.
      layer_elevation_point_is_inside = \
        np.min(layer_top_elevations_at_point[layers_above_point])
      layer_number = layer_numbers[layer_top_elevations_at_point \
                                   == layer_elevation_point_is_inside]
      if len(layer_number) > 1:
        sys.exit('entering too many layers!')
      else:
        layer_number = int(layer_number)
        #print "HERE HERE HERE"
        #print point,
        #print from_point
    #elif (point == from_point).all():
    else:
      print "*****************************"
      # Otherwise the point is on a layer top
      # STEP 1: FIND BOUNDARY LINE GOING THROUGH THIS POINT
      layer_top_numbers_at_point_elevation = self.layer_numbers[layer_tops_at_point_elevation]
      layer_bottom_numbers_at_point_elevation = self.layer_numbers[layer_bottoms_at_point_elevation]
      layer_numbers_at_point_elevation = self.layer_numbers[layers_at_point_elevation]
      # This helps if there are multiple layer tops at the layer elevation

      # Use angles
      # Line segment geometry
      C = point # center point
      # points around center
      L = [] # left sides of lines
      R = [] # right sides of lines
      for lnum in list(layer_top_numbers_at_point_elevation):
        l = self.layer_tops[lnum]
        vect = l - C
        vect_sign = np.sign(vect)
        # Is left (or on same point) if point is left or above (or equal
        # in each of these)
        isleft = (vect_sign[:,0] <= 0) * (vect_sign[:,1] >= 0)
        # otherwise is right
        isright = np.invert(isleft)
        #dist = self.layer_tops[ltnum]
        if isleft.any():
          L.append(l[isleft][-1,:]) # rightmost left point
        if isright.any():
          R.append(l[isright][0,:]) # leftmost right point
      for lnum in list(layer_bottom_numbers_at_point_elevation):
        l = self.layer_bottoms[lnum]
        vect = l - C
        vect_sign = np.sign(vect)
        # Is left (or on same point) if point is left or above (or equal
        # in each of these)
        isleft = (vect_sign[:,0] <= 0) * (vect_sign[:,1] >= 0)
        # otherwise is right
        isright = np.invert(isleft)
        #dist = self.layer_tops[ltnum]
        # Flip these at the end -- opposite orientation for bottom
        # Really CCW vs CW, not L vs R so much
        if isright.any():
          L.append(l[isright][0,:]) # rightmost left point, flipped
        if isleft.any():
          R.append(l[isleft][-1,:]) # leftmost right point, flipped
      L = np.array(L)
      R = np.array(R)
      # Check if any of these points is the center
      iscenter = np.prod(np.round(L, 5) == np.round(C, 5), axis=1, \
                                                           dtype=bool)
      # Convert to radial coordinates: arctan(y/x)
      Lrad = np.arctan2( L[:,1] - C[1], L[:,0] - C[0] )
      Lrad = np.array(Lrad)
      Lrad[Lrad<0] += 2*np.pi
      Lrad[iscenter] = np.nan # Remove from analysis
      Rrad = np.arctan2( R[:,1] - C[1], R[:,0] - C[0] )
      Rrad = np.array(Rrad)
      Rrad[Rrad<0] += 2*np.pi
      # Where does the line of erosion point
      Erad = np.arctan2( point[1] - from_point[1], point[0] - from_point[0])
      Erad = np.array(Erad)
      if Erad<0:
        Erad += 2*np.pi
      # Zones within units: CCW of L, CW of R
      # Inside the layer even if you're at its top
      # Not sure why -- sort of arbitrary
      # (and just changed it.)
      unit_number_inside = \
         layer_numbers_at_point_elevation[(Erad > Lrad) * (Erad <= Rrad)]
      if len(unit_number_inside) > 1:
        # pass and sort out later on?
        sys.exit("How are we inside multiple layers?")
      elif len(unit_number_inside) == 0:
        # About to enter free space
        # Follow the surface
        # And choose the leftmost if two come together
        #layer_number = -1
        #print "NO UNIT!"
        """
        try:
          layer_top_numbers_at_point_elevation = \
              layer_top_numbers_at_point_elevation[Lrad == np.nanmin(Lrad)][0]
        except:
          layer_top_numbers_at_point_elevation = \
              layer_top_numbers_at_point_elevation[Rrad == np.nanmin(Rrad)][0]
        """
        layer_number = -1
        #layer_number = [layer_top_numbers_at_point_elevation, 'entering free space']
        #layer_top_numbers_at_point_elevation = 0
      else:
        layer_top_numbers_at_point_elevation = unit_number_inside
        # Define layer_number
        layer_number = int(layer_top_numbers_at_point_elevation)
        #sys.exit("Knew it was possible to have >1 layer at a point but\n"+ \
        #         "did not yet prepare for it in the code.\n"+ \
        #         "Better do that now! [mid-layer]")
    
    print layer_number
    return layer_number
    
  def unique_rows(self, array):
      unique = np.ascontiguousarray(array).view(np.dtype((np.void, \
               array.dtype.itemsize * array.shape[1])))
      _, idx = np.unique(unique, return_index=True)
      layer = array[sorted(idx)]
      return layer
    
  def store_layers(self):
    """
    Save layers for visualization and analysis
    """
    pass

