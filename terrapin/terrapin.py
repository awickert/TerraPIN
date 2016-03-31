#! /user/bin/python

# Started 22 January 2015 by ADW
# based on some earlier notes and code

import numpy as np
from matplotlib import pyplot as plt
import sys

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
    print self.layer_tops
    print ""

  def updateFluvialTopo_y(self):
    """
    Lateral migration
    """
    pass
  
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
    from_point = point.copy()
    # "layer_updates" holds new points that modify layers until the end,
    # when we are ready to update the whole system all at once
    layer_updates = []
    chosen_layer_numbers = []
    #layer_updates.append(point)
    #chosen_layer_numbers.append(self.insideWhichLayer(point))
    topodefflag = False
    while point is not None:
      inLayer = self.insideOrEnteringWhichLayer(point)
      if np.prod(point == np.vstack(self.layer_tops), axis=1).any():
        # If this is the case, you are at some kind of intersecton.
        # Pick the layer immediately below to follow.
        # Got to make these rules more general sometime.
        # Leave boundary -- strange things happen on it.
        inLayer = self.insideOrEnteringWhichLayer([point[0]-1E-5, point[1]-1E5], self.layer_tops)
        #higher_points = self.layer_tops[inLayer]\
        #                [self.layer_tops[inLayer][:,1] > point[1]]
        # AT THIS POINT, JUST TAKE THE REST OF THE TOPOGRPAHY
        oldtopo = self.topo[self.topo[:,-1] > point[-1]]
        topo = np.vstack((oldtopo, np.vstack(layer_updates[::-1]), np.array([0, self.z_ch])))
        self.topo = topo.copy()
        topodefflag = True
        break
      elif inLayer is None:
        if point[1] < np.max(np.vstack(self.layer_tops)):
          # Must be along the top of an
          # internal layer that it has exited; go horizontally
          # until a new layer has been found
          # Start with last layer you were in; this will break
          # if this doesn't exist (starting above all layers -- should
          # aggrade instead in that case!)
          chosen_layer_number = chosen_layer_numbers[-1] # not even necessary -- still saved
          # If above all layers -- horizontally until it hits former topo
          newpoint = point.copy()
          newpoint[0] = self.piecewiseLinearAtZ(point[1], self.topo)
          chosenIntersection = newpoint
          chosen_layer_numbers.append(chosen_layer_number)
          layer_updates.append(chosenIntersection)
          # Now note that chosenIntersection is the new starting point
          from_point = point.copy()
          point = chosenIntersection.copy()
        else:
          print "Somehow your point is above the topography, while incising."
          break
      else:
        print "*", inLayer
        # slope-intercept
        angleOfRepose = self.alpha[self.layer_lithologies[inLayer]]
        m = - np.tan( (np.pi/180.) * angleOfRepose)
        b = point[1] - m*point[0]
        # Find intersection with each layer
        intersections = []
        for i in self.layer_numbers:
          # list of 1D numpy arrays
          intersections.append(self.findLikelyBestIntersection(m=m, b=b, \
                                    piecewiseLinear=self.layer_tops[i], \
                                    starting_point=point))
        intersections = np.squeeze(np.array(intersections))
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
          # And of these nonzero paths, find the shortest, and this is the
          # chosen intersection
          # Chosen layer number will work here because self.layer_numbers is
          # ordered just by a np.arange (so same)
          chosen_layer_number = (path_lengths == \
                                 np.nanmin(path_lengths)).nonzero()[0][0]
          chosenIntersection = intersections[chosen_layer_number]
          chosen_layer_numbers.append(chosen_layer_number)
          layer_updates.append(chosenIntersection)
          # Now note that chosenIntersection is the new starting point
          from_point = point.copy()
          point = chosenIntersection.copy()
    
    if topodefflag is False:
      # Wait until the end to update the cross-sectional profile
      for i in range(len(layer_updates)):
        layer_number = chosen_layer_numbers[i]
        intersection = layer_updates[i]
        # Then add this into the proper layer
        #print intersection
        self.layer_tops[layer_number] = np.vstack(( self.layer_tops[layer_number], np.expand_dims(intersection, 0) ))
        
      """
      for i in range(len(layer_updates)):
        self.layer_tops[i] = self.layer_tops[i][ self.layer_tops[i][:,0].argsort()]
      """
        
      # Sort it in order of increasing x so it is at the proper point
      for i in range(len(self.layer_tops)):
        self.layer_tops[i] = \
             self.layer_tops[i][ self.layer_tops[i][:,0].argsort()]

      # And after this, adjust the right-hand-sides of the layers to hit the river
      # NOT SURE THAT THIS IS REALLY NECESSARY -- LAYER VS. TOPOGRAPHIC SURFACE
      #if self.layer_tops[0][-1,-1] > self.z_ch:
      #  self.layer_tops[0][-1,-1] = self.z_ch
        # MAY HAVE TO ADD IN A SECOND POINT HERE ONCE TERRACES / LATERAL
        # MOTION COME ON LINE
      # NEED TO DEFINE TOPOGRAPHIC SURFACE SOMEWHERE
      # CHANNEL, VALLEY WALL, AND EACH FARTHEST RIGHT POINT ON EACH LAYER TOP

      intermediate_topo = self.newIncisedTopo(layer_updates)
      
      # Remove points that go beyond topo profile
      # Somehow inf points being removed here
      layer_top_index = 0
      for layer in self.layer_tops:
        row_indices = []
        row_index = 0
        for point in layer:
          x_point = point[0]
          y_point = point[1]
          y_topo = self.piecewiseLinearAtX(x_point, intermediate_topo)
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
        #if self.piecewiseLinearAtX(x_point, self.layer_tops[0]) < self.topo[-1][1]:
        self.layer_tops[i] = \
            np.append(self.layer_tops[i], 
                      np.expand_dims(self.layer_tops[i-1][-1], 0),
                      axis=0)
      #if self.piecewiseLinearAtX(x_point, self.layer_tops[0]) < self.topo[-1][1]:
      self.layer_tops[0] = \
          np.append(self.layer_tops[0], 
                    np.expand_dims(intermediate_topo[-1], 0),
                    axis=0)
      
      print intermediate_topo
      self.topo = intermediate_topo[:]
      
      # Probably unnecessary
      self.layer_tops = self.rmdup(self.layer_tops)

  def rmdup(self, layers):
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
    x_valley_wall = self.piecewiseLinearAtZ(self.z_ch, self.topo)
    aggraded_surface = np.array([[x_valley_wall, self.z_ch],
                                 [0, self.z_ch]])
    oldtopo = self.topo[self.topo[:,-1] > self.z_ch]
    topo = np.vstack((oldtopo, aggraded_surface))
    self.topo = topo.copy()

    layer_is_alluvium = np.array(self.layer_lithologies) == 'alluvium'
    # Need to specify axis -- DEPRECATION WARNING
    alluv_layers = list(np.array(self.layer_tops)[layer_is_alluvium])
    alluv_layer_numbers = self.layer_numbers[layer_is_alluvium]
    
    # And then see if any of this is alluvium
    # And/or see where alluvium is
    # SPACE HERE TO ADD A NEW LAYER OR INCORPORATE IT INTO OTHERS
    # top one should never work.
    contacts_layer_number = self.insideOrEnteringWhichLayer \
                                                 (aggraded_surface[0], \
                                                  aggraded_surface[0])
    # but this might
    if contacts_layer_number is None:
      layers_below = []
      for layer in self.layer_tops:
        layers_below.append(self.piecewiseLinearAtX(0, layer))
      layer_number_immediately_below = (layers_below == \
                                    np.nanmax(layers_below)).nonzero()[0][0]
      # REDUNDANT CHECK
      if self.layer_lithologies[layer_number_immediately_below] == 'alluvium':
        contacts_layer_number = layer_number_immediately_below
    tmplayer = None
    # This will work only if alluvium is in valley, not on broader surface
    if contacts_layer_number: # Are we inside any layer?
      if self.layer_lithologies[contacts_layer_number] == 'alluvium':
        tmplayer = self.layer_tops[contacts_layer_number][:]
      # If it intersects here, then new layer is below
      tmplayer = tmplayer[tmplayer[:,1] > aggraded_surface[0,1]]
    else:
      tmp_layer_tops = self.layer_tops[:]
      tmp_layer_tops.append(aggraded_surface)
      tmp_layer_numbers = np.arange(len(self.layer_numbers)+1)
      for i in range(len(alluv_layer_numbers)):
        alluv_layer = alluv_layers[i]
        alluv_layer_number = alluv_layer_numbers[i]
        from_point = alluv_layer[0].copy() # starting
        for point in alluv_layer:
          inlayer = self.insideOrEnteringWhichLayer(point=point, \
                                          from_point=from_point, \
                                          layers=tmp_layer_tops, \
                                          layer_numbers=tmp_layer_numbers)
          if inlayer == tmp_layer_numbers[-1]:
            contacts_layer_number = alluv_layer_number
            tmplayer = self.layer_tops[contacts_layer_number][:]
          from_point = point.copy()
      # In this case, layer is above -- can simply append
      # though x will make a jog back to the left, to the surprise of all!
      # (via valley geometry)
    if tmplayer is not None:
      # COMBINE THIS WITH OTHER ALLUV LAYER
      tmplayer = np.vstack((tmplayer, aggraded_surface))
      tmplayer = tmplayer[tmplayer[:,-1] >= self.z_ch] # take only points
                                                       # not buried under alluv
      self.layer_tops[contacts_layer_number] = tmplayer[:]
    else:
      self.layer_tops = tmp_layer_tops[:]
      self.layer_numbers = tmp_layer_numbers[:]
      self.layer_lithologies.append('alluvium')
      
    # Now have diff. topo fcn for aggrading -- need to have 2x fcn.'s?
    #self.topographicProfile(self.layer_tops)
  
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
    topo = self.rmdup(topo)
    topo = topo[ topo[:,0].argsort()]
    # Not final topo -- intermediate step.
    # So don't update self.topo
    return topo
  
  def topographicProfile(self, layers):
    # Topographic profile
    # Pick only the highest points at each position
    topo = []
    #allpoints = np.concatenate(layers[::-1])
    topoPoints = []
    for layer in layers:
      layerPoints = []
      for point in layer:
        #print point
        layer_elevations_at_x = []
        for layer in layers:
          layer_elevations_at_x.append( self.piecewiseLinearAtX(point[0], layer) )
        layer_elevations_at_x = np.array(layer_elevations_at_x)
        layer_elevations_at_x = \
              layer_elevations_at_x[np.isnan(layer_elevations_at_x) == False]
        if (np.round(point[1], 10) >= np.round(np.array(layer_elevations_at_x), 10)).all():
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
      #print layerPoints.ndim == 1
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
    topo = self.rmdup(topo)
    topo = topo[ topo[:,0].argsort()]
    # Not final topo -- intermediate step.
    # So don't update self.topo
    return topo
    
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
    #print kwargs
    # Should make ylim 0 at start to do this properly
    topoFinite = self.topo.copy()
    #print topoFinite
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
  
  def layerPlot(self)
    fig, ax = plt.subplots()
    points = np.vstack(self.layers)
    minx_not_inf = np.min(points[np.isinf(points[:,0]) == False][:,0])
    infinity_to_left = minx_not_inf*2 - 1
    i=0
    for layer in self.layers:
      layer[:,0][np.isinf(layer[:,0])] = infinity_to_left
      shape = plt.Polygon(layer, label=self.layer_lithologies[i])
      ax.add_patch(shape)
      i+=1
    plt.axis('scaled')
    plt.legend()
    #labels = self.layer_lithologies
    #legend = plt.legend(labels, loc=(0.9, .95), labelspacing=0.1)
    #plt.setp(legend.get_texts(), fontsize='small')
  
  def layer_boundaries(self):
    """
    For each line segment:
      find all highest points that are below it.
      append these to a big line.
    Attach this to the layer_top stuff, as the layer bottom
    And then there is a closed boundary, and I can write methods to test what
    touches this or is inside it.
    """
    self.layers = []
    for layer_top in self.layer_tops:
      highest_points_below = self.highest_points_below(layer_top)
      # Reverse order to go from right to left -- make loop
      layer = np.vstack((layer_top, highest_points_below[::-1]))
      self.layers.append(layer)
  
  def highest_points_below(self, pwl):
    
    highest_points_below = []
    left = pwl[0,0]
    right = pwl[-1,0]
    layer_top_points = np.vstack(self.layer_tops)
    layer_top_points = layer_top_points[(layer_top_points[:,0] >= left) *
                                        (layer_top_points[:,0] <= right)]
    for point in layer_top_points:
      ztop = self.piecewiseLinearAtX(point[0], pwl)
      if point[1] < ztop:
        highest_points_below.append(point)
    highest_points_below = self.unique_rows(np.vstack(highest_points_below))
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
      if (xint >= xy0[0]) and (xint <= xy1[0]):
        # You're done!
        # y-value plugging into one equations -- let's use the line from the 
        # starting point that is producing the erosion
        yint = m*xint + b
        # Because there is some numerical error created by defining yint
        # with such an equation let's add a rounding term: round to nearest
        # 1E-9 m (nanometer) -- because totally insignificant in these systems
        intersections.append(np.round(np.array([xint, yint]), 9))
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
      z = np.nan
    else:
      # First, define line segment of interest
      xmin_pwl = np.max( pwl[:,0][pwl[:,0] <= x] )
      xmax_pwl = np.min( pwl[:,0][pwl[:,0] >= x] )
      z_xmin_pwl = np.mean(pwl[:,1][pwl[:,0] == xmin_pwl]) # in case two have it
      z_xmax_pwl = np.mean(pwl[:,1][pwl[:,0] == xmax_pwl]) # in case two have it
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
  
  def insideOrEnteringWhichLayer(self, point, from_point, layers=None, layer_numbers=None):
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
    
    if layers is None:
      layers = self.layer_tops
    if layer_numbers is None:
      layer_numbers=self.layer_numbers
    
    if type(layers) == np.ndarray:
      layers = [layers]
    
    layer_elevations_at_point = []
    for i in range(len(layers)):
      layer_elevations_at_point.append(self.piecewiseLinearAtX(point[0], layers[i]))
    layer_elevations_at_point = np.array(layer_elevations_at_point)
    
    if (layer_elevations_at_point != point[1]).all():
      # If all layer tops are above or below the point, the corresponding
      # layer top must be the next one above the point.
      # 
      # If there is no such layer top, then the point is in free space.
      #
      # Get invalid value error if there is a nan, which means that the layer
      # does not exist above that point
      # But this will always be false anyway, so this is fine. Just suppress
      # the error
      with np.errstate(invalid='ignore'):
        layers_above_point = layer_elevations_at_point > point[1]
      if layers_above_point.any():
        layer_elevation_point_is_inside = \
          np.min(layer_elevations_at_point[layers_above_point])
        layer_number = layer_numbers[layer_elevations_at_point \
                                     == layer_elevation_point_is_inside]
        layer_number = int(np.mean(layer_number))
      else:
        # If neither of these, must be above everything
        layer_elevation_point_is_inside = None
        layer_number = None
    # Otherwise the point is on a layer
    layers_at_point_elevation = (layer_elevations_at_point == point[1])
    if point == from_point:
      # If this is the case, then we are at the first point in the layer.
      # The layer top exists at our position.
      with np.errstate(invalid='ignore'):
        # There should be only one layer at this point's x, z position
        layer_number = self.layer_numbers[layers_at_point_elevation]
        try:
          layer_number = int(np.mean(layer_number))
        except:
          sys.exit("Knew it was possible to have >1 layer at a point but\n"+ \
                   "did not yet prepare for it in the code.\n"+ \
                   "Better do that now! [starting point]")
    else:
      # In a more general case, we need to consider the point that we are coming
      # from, and which layer we are entering.
      # This is because the layer being entered determines the next angle eroded
      # through the layer, which is the main point (at least for now) of knowing
      # which layer we are in -- so perhaps this should be changed to "in" or 
      # "entering" in the function title
      # Find layer boundary line going through this point,
      # find angle of incidence of incoming line,
      # and then find layer that is entered
      # STEP 1: FIND BOUNDARY LINE GOING THROUGH THIS POINT
      layer_top_number_at_point_elevation = self.layer_numbers[layers_at_point_elevation]
      if len(layer_top_number_at_point_elevation > 0):
        sys.exit("Knew it was possible to have >1 layer at a point but\n"+ \
                 "did not yet prepare for it in the code.\n"+ \
                 "Better do that now! [mid-layer]")
      lt = self.layer_tops[layer_top_number_at_point_elevation]
      ltx = lt[:,0]
      lty = lt[:,1]
      qualified_vertices = (ltx < point[0])
      if qualified_vertices.any():
        left = lt[ltx == np.max(ltx[qualified_vertices]),:]
      else:
        # If this fails, means there is no point <, so try points =
        (ltx == point[0]) * (lty > point[1])
        if qualified_vertices.any():
          left = lt[lty == np.min(lty[qualified_vertices])]
        else:
          sys.exit("Is this layer a lens that doesn't go to -inf in x?\n"+
                   "This has not been tested yet, so test and then remove\n"+
                   "this line when you know that all is working.")
      # Find the "right" as the vertex that is just next after "left"
      # If this doesn't work, whole sorting system has gone down!
      left_i = np.prod(lt == left, axis=1).nonzero()[0][0]
      right_i = left_i + 1
      right = lt[right_i]
      # Step 2: Find slopes.
      slope_layer_top = (right[1] - left[1]) / (right[0] - left[0])
      slope_line_to_point = (point[1] - from_point[1]) / \
                             point[0] - from_point[0])
      # Step 3: Use slope comparison to decide which layer to choose
      # (Note: If only 2 layers always meet, I could have circumvented this
      #  by simply looking at the layer in which the origin lay, and choosing
      #  the other layer)
      if slope_layer_top < slope_line_to_point:
        # If layer top decreases more steeply than line intersecting it, look
        # below the line.
        layers_below = layer_elevations_at_point[ \
                       (layer_elevations_at_point < point[0]) ]
        highest_layer_below = layer_elevations_at_point == np.max(layers_below)
        layer_number = self.layer_numbers[highest_layer_below]
      else:
        # (even if slope_layer_top == slope_line_to_point)
        # Look above line: pick layer top
        layer_number = self.layer_numbers[layers_at_point_elevation]
      # Step 2: find which layers meet here
      #self.layers == points.left.any()
      
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

