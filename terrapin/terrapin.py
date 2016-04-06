#! /user/bin/python

# Started 22 January 2015 by ADW
# based on some earlier notes and code

import numpy as np
from matplotlib import pyplot as plt
import sys
import fnmatch

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
    for layer_top in self.layer_tops:
      print layer_top
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
    from_point = None # NULL in future
    # "topo_updates" holds new points that modify layers until the end,
    # when we are ready to update the whole system all at once
    topo_updates = []
    chosen_layer_numbers = []
    #chosen_layer_numbers.append(self.insideWhichLayer(point))
    topodefflag = False
    while point is not None:
      inLayer = self.insideOrEnteringWhichLayer(point, from_point)
      from_point = point.copy()
      if inLayer is None:
        # Above all? Then should be at the top.
        # After going above topographic surface, nothing to do.
        point = None # break out of loop.
      else:
        # All is normal -- carry on, cowboy/girl!
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
          topo_updates.append(chosenIntersection)
          # Now note that chosenIntersection is the new starting point
          point = chosenIntersection.copy()
    
    if topodefflag is False:
      # Wait until the end to update the cross-sectional profile

      self.topo = self.newIncisedTopo(topo_updates)
      
      # Completely remove all layers that lie entirely above topography
      self.layer_boundaries = self.calc_layer_boundaries() # refresh      
      for n in range(len(self.layer_boundaries)):
        layer = self.layer_boundaries[n]
        topo_at_layer_points = []
        for point in layer:
          topo_at_layer_points.append(self.piecewiseLinearAtX(point[0], self.topo))
        topo_at_layer_points = np.array(topo_at_layer_points)
        if (topo_at_layer_points < layer[:,1]).all():
          print n
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
                print topo_point
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
          y_topo = self.piecewiseLinearAtX(x_layer_top, self.topo)
          if y_topo >= y_layer_top:
            #print point
            layer_top_trimmed.append(point)
        print np.array(layer_top_trimmed)
        self.layer_tops[i] = np.array(layer_top_trimmed)

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
      self.layer_tops = self.rmdup(self.layer_tops)
      
      # Then sort it all
      for i in range(len(self.layer_tops)):
        layer_top = self.layer_tops[i]
        self.layer_tops[i] = layer_top[layer_top[:,0].argsort()]

      # NOT SURE IF THIS IS NEEDED!
      # Then check for unnecesssary points and remove them
      for i in range(len(self.layer_tops)):
        layer_top = self.layer_tops[i]
        slopes = np.diff(layer_top[:,1]) / np.diff(layer_top[:,0])
        # Those with constant slopes on both sides are not needed
        not_needed = np.hstack(( False, slopes[:-1] == slopes[1:], False ))
        self.layer_tops[i] = self.layer_tops[i][not_needed == False]
      

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
    print ""
    print touching_layers
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
        gt_cond = point[1] >= self.piecewiseLinearAtX(point[0], 
                                                      self.layer_tops[n])
        nan_cond = np.isnan( self.piecewiseLinearAtX(point[0], 
                                                      self.layer_tops[n]) )
        if gt_cond or nan_cond:
          point_geq_layer.append(True)
        else:
          point_geq_layer.append(False)
      if np.array(point_geq_layer).all():
        print point
        combined_layer_top.append(point)
    combined_layer_top = np.array(combined_layer_top)
    combined_layer_top = self.rmdup(combined_layer_top)
        
    # Remove layers with higher layer numbers
    for n in touching_alluvial_layer_numbers[touching_alluvial_layer_numbers!=
                                     np.min(touching_alluvial_layer_numbers)]:
      self.layer_numbers = np.delete(self.layer_numbers, n)
      self.layer_names.pop(n)
      self.layer_lithologies.pop(n)
      self.layer_tops.pop(n)
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
    topo = self.rmdup(topo)
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
        if (np.round(point[1], 6) >= np.round(np.array(layer_elevations_at_x), 6)).all():
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
    print 'layers\n',layers
    print '***', np.max(layers[:,0]), np.min(layers[:,0])
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
    print '***', topo

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
    topo = self.rmdup(topo)
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
  
  def layerPlot(self):
    fig, ax = plt.subplots()
    layers = self.calc_layer_boundaries()
    points = np.vstack(layers)
    minx_not_inf = np.min(points[np.isinf(points[:,0]) == False][:,0])
    infinity_to_left = minx_not_inf*2 - 1
    miny_not_inf = np.min(points[np.isinf(points[:,1]) == False][:,1])
    infinity_to_bottom = miny_not_inf*1.2 - 1
    i=0
    color_cycle = ['r', 'g', 'b', 'c', 'm', 'y']
    for layer in layers:
      layer[:,0][np.isinf(layer[:,0])] = infinity_to_left
      layer[:,1][np.isinf(layer[:,1])] = infinity_to_bottom
      shape = plt.Polygon(layer, facecolor=color_cycle[i%len(color_cycle)], edgecolor='k', label=self.layer_lithologies[i])
      ax.add_patch(shape)
      i+=1
    # plotting limits
    all_points = np.vstack(layers)
    plt.xlim(( np.min(all_points[:,0]), np.max(all_points[:,0]) ))
    plt.ylim(( np.min(all_points[:,1]), np.max(all_points[:,1]) ))
    #plt.axis('scaled')
    plt.legend(loc='bottom left')
    #plt.show()
    #labels = self.layer_lithologies
    #legend = plt.legend(labels, loc=(0.9, .95), labelspacing=0.1)
    #plt.setp(legend.get_texts(), fontsize='small')
  
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
                           self.piecewiseLinearAtX(point[0], other_layer), 6))
        if (np.round(point[1], 6) > \
                   np.array(other_layer_tops_at_point)).all():
          # Although the first cut looked for layers below, not every layer is
          # completely below.
          # So check that all new points are below.
          print point, self.piecewiseLinearAtX(point[0], layer_top)
          if np.round(point[1], 6) < \
                   np.round(self.piecewiseLinearAtX(point[0], layer_top), 6):
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
    layer_bottoms = []
    for i in range(len(self.layer_tops)):
      layer_bottoms.append(self.calc_layer_bottom(self.layer_boundaries[i],
                                                  self.layer_tops[i]))

    return layer_bottoms
  
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
      if (xint >= xy0[0]) and (xint <= xy1[0]):
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
        #intersections.append(np.round(np.array([xint, yint]), 6))
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
    
    return z[0]
    
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
  
  def insideOrEnteringWhichLayer(self, point, from_point=None, layers=None, layer_numbers=None):
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
    
    print 'point', point

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
      layers_above_point = layer_elevations_at_point > point[1]
      point_above_all_layers = (layer_elevations_at_point < point[1]).all()
    print layer_elevations_at_point
    print point[1]
    print (layer_elevations_at_point != point[1]).all()
    if point_above_all_layers:
      # Must be above everything, the point is in free space!
      layer_elevation_point_is_inside = None
      layer_number = None
      print "Warning: POINT ABOVE ALL LAYERS!"
    # 2. This should also work if it is the first point.
    #    Check if this is the first point in the series
    elif (np.round(layer_elevations_at_point, 6) != np.round(point[1], 6)).all():
      # Point does not lie on a layer top, but there should be at least
      # one layer above it.
      layer_elevation_point_is_inside = \
        np.min(layer_elevations_at_point[layers_above_point])
      layer_number = layer_numbers[layer_elevations_at_point \
                                   == layer_elevation_point_is_inside]
      if len(layer_number) > 1:
        sys.exit('entering too many layers!')
      else:
        layer_number = int(layer_number)
        print "HERE HERE HERE"
        print point,
        print from_point
    #elif (point == from_point).all():
    else:
      print "*****************************"
      # Otherwise the point is on a layer
      layers_at_point_elevation = (np.round(layer_elevations_at_point, 6)
                                            == np.round(point[1], 6))
      """
      if (point == from_point).all():
        # If this is the case, then we are at the first point in the layer.
        # The layer top exists at our position.
        with np.errstate(invalid='ignore'):
          # There should be only one layer at this point's x, z position
          layer_number = self.layer_numbers[layers_at_point_elevation]
          try:
            layer_number = int(layer_number)
          except:
            sys.exit("Knew it was possible to have >1 layer at a point but\n"+ \
                     "did not yet prepare for it in the code.\n"+ \
                     "Better do that now! [starting point]")
      else:
      """
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
      if len(layer_top_number_at_point_elevation) > 1:
        # band-aid
        print 'FROM', from_point
        if (from_point == np.array([0, self.z_ch])).all() and \
           (layer_top_number_at_point_elevation == \
           (len(self.layer_numbers)-1)).any():
          layer_top_number_at_point_elevation = \
                      np.min(layer_top_number_at_point_elevation)
          layers_at_point_elevation[-1] = False
        else:
          sys.exit("Knew it was possible to have >1 layer at a point but\n"+ \
                   "did not yet prepare for it in the code.\n"+ \
                   "Better do that now! [mid-layer]")
      lt = self.layer_tops[layer_top_number_at_point_elevation]
      ltx = lt[:,0]
      lty = lt[:,1]
      qualified_vertices = (ltx < point[0])
      self.ltx = ltx
      self.lty = lty
      self.qualified_vertices = qualified_vertices
      self.point = point
      if qualified_vertices.any():
        left = np.squeeze(lt[ltx == np.max(ltx[qualified_vertices]),:])
      else:
        # If this fails, means there is no point <, so try points =
        (ltx == point[0]) * (lty > point[1])
        if qualified_vertices.any():
          left = np.squeeze(lt[lty == np.min(lty[qualified_vertices])])
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
      print 'from_point', from_point
      if from_point is not None:
        slope_layer_top = (right[1] - left[1]) / (right[0] - left[0])
        if (point[0] - from_point[0]) == 0:
          slope_line_to_point = -np.inf
        else:
          slope_line_to_point = (point[1] - from_point[1]) / \
                                (point[0] - from_point[0])
      else:
        sys.exit("A from_point is needed here")
      # Step 3: Use slope comparison to decide which layer to choose
      # (Note: If only 2 layers always meet, I could have circumvented this
      #  by simply looking at the layer in which the origin lay, and choosing
      #  the other layer)
      print 'slopes',
      print slope_layer_top,
      print slope_line_to_point
      if slope_layer_top < slope_line_to_point:
        # Look below line: pick layer top
        #print layers_at_point_elevation
        layer_number = self.layer_numbers[layers_at_point_elevation]
      else:
        # If layer top decreases more steeply than line intersecting it, look
        # below the line.
        layers_above = layer_elevations_at_point[
                       layer_elevations_at_point > point[1] ]
        if len(layers_above) == 0:
          layer_number = None # entering free space -- other code should
                              # take this along the top of the domain
        else:
          lowest_layer_above = layer_elevations_at_point == np.min(layers_above)
          layer_number = int(self.layer_numbers[lowest_layer_above])
          # (even if slope_layer_top == slope_line_to_point)
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

