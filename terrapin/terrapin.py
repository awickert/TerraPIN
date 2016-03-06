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

  def update(self):
    # no alluviation yet
    self.incise()
    self.erode_laterally()

  def finalize(self):
    pass

  # FUNCTIONS TO HANDLE PARTS OF THE RUNNING
  def set_input_values(self):
    """
    Eventually will set input variable values in some more end-user-interactive
    way, but currently just is where I type the values into this file itself.
    Early stages of development!
    """
    # elevation of the channel bedrock surface at the starting time
    self.z_br_ch = -60.
    # ASSUMING FOR NOW THAT IT STARTS ON A FLAT PLANE, AND SOMEHOW IS NOT
    # CAUSING AN ERROR -- ACTUALLY, THIS WILL BE SOMETHING ON THE TO-DO LIST:
    # NORMALLY HAVE UNIFORM AGGRADATION ACROSS VALLEY, BUT ALLOW IT TO GO OVER 
    # SOME MAXIMUM DISTANCE IF VALLEY BECOMES TOO WIDE -- PERHAPS RELATED TO 
    # REWORKING RATE OF SYSTEM AND TIME STEP
    # Angle of repose of alluvium
    alpha_a = 32.
    # Angle of repose of bedrock 
    alpha_r = 75.
    # PROBABLY PUT THESE TWO IN A LIST TO WRITE MORE GENERAL CODE THAT CAN 
    # CYCLE THROUGH ANY NUMBER OF LAYER AND THEIR PARAMETERS
    # NOW SOME GARBAGE CAN COEFFICIENT THAT COMBINES ERODIBILITY AND HOW 
    # HARD IT IS TO REMOVE THE MATERIAL (PROBABLY RELATED TO HOW LARGE 
    # THE RESULTING BLOCKS WILL BE). THEREFORE THIS SHOULD BE RELATED IN 
    # SOME WAY TO THE ANGLE OF REPOSE (FRICTION ANGLE) BUT WILL LEAVE 
    # SEPARATE FOR NOW.
    # UNITS WILL DEPEND ON FORMULATION, BUT RIGHT NOW WILL BE [LENGTH^2/TIME]
    # Erodibility coefficient of alluvium
    k_a = 1E-2
    # Erodibility coefficient of bedrock
    k_r = 1E-4
    # Surface elevation profiles -- start out flat
    # bedrock (x,z)
    z_br = np.array([[-np.inf, 0], [0, 0]])
    # sediment (x,z)
    # HAVE TO DO SOMETHING ELSE IF NO SED PRESENT -- JUST TAKE IT OUT OF
    # ARRAY, MOST LIKELY.
    #z_sed = np.array([[-np.inf, -np.inf], [0, -np.inf]])
    # Define channel width at this point
    self.b = 50 # [m]
    # And the channel will start out the same width as its bedrock-valley
    self.B = 50 # [m]
    
    # Create arrays of values of angles and resistance to erosion
    # Alluvium is always the last one
    # Goes from bottom of strat column to top
    #self.alpha = [40., alpha_r, alpha_a]
    #self.k = [1E-2, k_r, k_a]
    #self.layer_tops = [z_br, z_sed] # Just temporary/arbitrary
    #self.layer_names = ['bedrock', 'alluvium']
    self.alpha = [alpha_r]
    self.k = [k_r]
    self.layer_tops = [z_br]
    self.layer_names = ['bedrock']
    self.layer_numbers = np.arange(len(self.layer_tops))
    """
    self.alpha = [alpha_r, alpha_a]
    self.k = [k_r, k_a]
    self.layer_tops = [z_br, z_sed]
    self.layer_names = ['bedrock', 'alluvium']
    self.layer_numbers = np.arange(len(self.layer_tops))
    """
    
  # MUST acknowledge off-calculation finite channel width, b: valley
  # width *never* 0. 
  
  # Then can have functions for the differnet options on how to do things
  # like:
  
  def Channel_Touching_Side_Braided_Random_Position(self):
    """
    After Wickert et al. (2013), JGR.
    Channel position in cross-stream sense is random
    (see also note by Bradley and Tucker -- small valley width, this is 
    OK for meandering streams too)
    """
    # Remember that valley width = 2x what we have on model, + b
    # B_mod = flat valley bottom width -- get this somehow -- what about 
    # gradual incision issues though -- should keep affecting walls for some time!
    # Yeah, lateral migration into sloping surface issue.
    B = 2*B_mod + b
    # So always touching wall when B = b, no excess valley width
    self.Pch = b/B


  def erode_laterally(self):
    # Might get complicated when everything isn't at just one elevation, h,
    # or isn't just one material. In fact, will have to iterate, I think.
    # But for now, keeping it simple
    point = np.array([0, self.z_br_ch]) # Fix this in future to get edge of 
                                        # valley wall
    inLayer = self.insideWhichLayer(point)
    # Valley width and probability of touching valley wall
    B = 2*B_mod + b
    Pch = b/B
    # Hm, I think I might stop here. Lateral erosion will require the same
    # get-point-and shoot tools as vertical erosion (so worth generalizing
    # that), plus the possibility of iteration to an appropriate solution
    # considering all the debris above -- and also thinking about how to 
    # handle that
    # So try multiple bedrock layers first, how about that?



  # For width relationship, Wickert et al. (2013) found that on braidplains
  # with no internal terraces (these presumed to cauase deviations from fit),
  # river position must be random b/c exponential decay shape to pixels 
  # visited. (Not so for meandering case: is power law around center -- 
  # see Nate Bradley and Greg Tucker's paper.
  # So I can assume braided and say contact with wall goes as 
  # 1/(b-B) or something like this, but noting that b does go up (though 
  # therefore power at contact goes down -- ooh, write an equation for this 
  # too) 
  # And then lower chance of contact with wall for meandering? How long will
  # channel stay against walls -- will it be pushed against them b/c it wants 
  # to have a more free floodplain? (Personifying, I know :) )
  # Check Nate's paper. For now, use my braided approx.

    







  def updateTopo(self):
    # Will soon hold much of the "incise" machinery
    # because "intersection" list is really topo, and this will then be mapped
    # onto the layers
    pass




  # START OUT BY WRITING SEPARATE INCISION AND AGGRADATION ALGORITHMS FOR 
  # SIMPLICITY. MAY EVENTUALLY BECOME PART OF THE SAME FUNCTION, OR AT LEAST
  # SHARE SOME METHODS
  def incise(self):
    """
    When river incises, compute erosion and angle-of-repose destruction of
    material above.
    When collapse of bedrock does happen, it creates an angle-of-repose pile
    of sediments in the real world. Represent this in some way?
    """
    # THINK THAT IF I CHANGE THE POINT HERE, I CAN GENERALIZE THIS TO LATERAL EROSION OF TERRACES AS WELL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # BUT THEN WOULD NEED TO REMOVE THE IMMEDIATE UPDATING STEP TO ADD AN ITERATION FOR LAYERS ABOVE THE LAYER IN QUESTION -- DIFFERENT ERODIBILITIES AND SLOPE LENGTHS
    # AND/OR I WONDER IF IT WOULD BE POSSIBLE TO PREEMPTIVELY SUM THE ERODIBILITIES 
    # AND SLOPE LENGTHS OF ALL LAYERS ABOVE A POINT -- SLOPE LENGTHS YES, BUT IF LAYERS 
    # ARE NON-HORIZONTAL AT THEIR BASE (E.G., ALLUVIUM), THEN WOULD HAVE TO ITERATE ANYWAY
    point = np.array([0, self.z_br_ch])
    print "*", self.insideWhichLayer(point)
    # And layer_updates holds new points that modify layers until the end,
    # when we are ready to update the whole system all at once
    layer_updates = []
    while point is not None:
      inLayer = self.insideWhichLayer(point)
      print inLayer
      # if inLayer is none, it must be above all layers.
      # then break out of loop at end
      # POSSIBLE THAT PROBLEM WILL BE CAUSED BY HAVING POINTS ALSO BE ABLE TO
      # BE ON TOP OF LAYERS
      if inLayer is not None:
        #if len(layer_updates) == 0:
        #  layer_updates.append([inLayer, point.copy()])
          #layer_updates[0][1][1] += 1
        angleOfRepose = self.alpha[inLayer]
        # Slope -- minus because solving for what is left of river
        m = - np.tan( (np.pi/180.) * angleOfRepose)
        # Intercept
        b = point[1] - m*point[0]
        # Find intersection with each layer
        intersection = []
        for i in self.layer_numbers:
          # list of 1D numpy arrays
          intersection.append(self.findIntersection(m, b, self.layer_tops[i]))
        # turn it into a 2D array from a 1D list of 1D arrays
        intersection = np.array(intersection)
        # Define the chosen intersection
        chosen_intersection = intersection.copy()
        # First, use only those points are above the point in question
        intersection[intersection[:,1] <= point[1]] = np.nan
        # if nothing above, then we are at the top
        if np.isnan(intersection).all():
          # Break out of loop
          point = None
        else:
          # Find the path lengths to these intersections
          path_lengths = ( (intersection[:,0] - point[0])**2 \
                         + (intersection[:,1] - point[1])**2 )
          # And of these nonzero paths, find the shortest, and this is the
          # chosen intersection
          # Chosen layer number will work here because self.layer_numbers is
          # ordered just by a np.arange (so same)
          chosen_layer_number = (path_lengths == np.nanmin(path_lengths)).nonzero()[0][0]
          chosen_intersection = intersection[chosen_layer_number]
          layer_updates.append([chosen_layer_number, chosen_intersection])
          # Now note that chosen_intersection is the new starting point
          point = chosen_intersection.copy()
      point = None
    
    # Wait until the end to update the cross-sectional profile
    for i in range(len(layer_updates)):
      number = layer_updates[i][0]
      intersect = layer_updates[i][1]
      # Then add this into the proper layer
      #print self.layer_tops[number]
      self.layer_tops[number] = np.vstack(( self.layer_tops[number], np.expand_dims(intersect, 0) ))
      #print self.layer_tops[number]
      # And remove those values above it -- vertices from pre-incision topography
      # Each intersect is origin for next shot, in turn, so that makes things easier.
      # Can't use river as intersect #1 because can't calculate an appropriate slope
      #if i == 0:
      #  self.removeOldVertices(np.array([0, self.z_br_ch]), intersect, number)
      #else:
      #  self.removeOldVertices(layer_updates[i-1][1], intersect, number)
      if i > 0:
        self.removeOldVertices(layer_updates[i-1][1], intersect, number)
      else:
        pass
        #WORK HERE!!!
      # Remove duplicates too, such as those that appear when there is 
      # neither incision nor aggradation
      # Following the answer at:
      # http://stackoverflow.com/questions/8560440/removing-duplicate-columns-and-rows-from-a-numpy-2d-array
      unique = np.unique(self.layer_tops[number].view([('', \
               self.layer_tops[number].dtype)]*self.layer_tops[number].shape[1]))
      self.layer_tops[number] = unique.view(self.layer_tops[number].dtype).reshape((unique.shape[0], self.layer_tops[number].shape[1]))
     
    # And sort it in order of increasing x so it is at the proper point
    for i in range(len(layer_updates)):
      self.layer_tops[i] = self.layer_tops[i][ self.layer_tops[i][:,0].argsort()]
      # And after this, adjust the right-hand-sides of the layers to hit the river
      self.cutLayersToRiver()
    
    for i in range(len(self.layer_tops)):
      print ""
      print self.layer_tops[i]
    print ""
    print "================="
    
    # NEXT NEED TO REMOVE REDUNDANT VERTICES
    # self.removeOldVertices (above) not working
  
  def cutLayersToRiver(self):
    """
    Ensure that the layers drop into the river.
    """
    for layer in self.layer_tops:
      if layer[-1,-1] > self.z_br_ch:
        layer[-1,-1] = self.z_br_ch
  
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

  def aggrade(self):
    """
    Fill up area within the valley with sediment (or disperse sediment 
    over some distance on a flat plain? Or just error/quit in that case?
    """
    # Treat alluvium aggradation as distinctly different: just fills space
    # up to certain level. If < max elevation of alluvium, draw a new line 
    # (horizontal) and snap new points... otherwise it will be just 
    pass
    
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
      # In case the point is at intersection of two segments, just give it 
      # the known z-value
      # (this could have been avoided by using < and >= instead of <= and >=,
      # but then that could cause problems of the point to be selected were 
      # on the edge of the domain but still hanging on to the last defined 
      # point
      if xmin_pwl == xmax_pwl:
        # In this case, z_xmin_pwl and z_xmax_pwl are defined to be the same,
        # so just pick one
        z = z_xmin_pwl
      else:
        # define the line given by the segment endpoints
        # z = mx + b 
        # slope
        m = (z_xmax_pwl - z_xmin_pwl)/(xmax_pwl - xmin_pwl)
        # intercept -- could equally calculate with x_max
        b = z_xmin_pwl - m*xmin_pwl
        # Now can compute z
        z = m*x + b
      
    return z
    
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
    layers_above_point = layer_elevations_at_point > point[1]
    if layers_above_point.any():
      layer_elevation_point_is_inside = layer_elevations_at_point[layers_above_point]
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
    
    
  def removeRedundantVertices(self):
    """
    Think this will test slope on either side of each vertex, and if they
    are the same, will remove it.
    Not entirely good -- doesn't remove proper point on incision
    """
    for layer_top in self.layer_tops:
      slopes = np.diff(layer_top[:,1])/np.diff(layer_top[:,0])
      redundant_segments = np.diff(slopes) == 0
      # First point and last point are never redundant, so this will
      # be for one of the midpoints, hence has length - 2 when compared
      # to t.layer_tops
      new_layer_top = np.vstack(( layer_top[0,:],
                                  layer_top[1:-1,:][redundant_segments == False],
                                  layer_top[-1,:] ))
    
    
  
  #self.removeOldVertices(layer_updates[i][1], intersect, number)
        
  def removeOldVertices(self, origin, intersection, layer_number):
    """
    This will then find out if there are old vertices that lie above the new
    line. Just have to pass new line endpoints to it and the characteristics of 
    the 
    
    intersection = new endpoint computed (x,z)
    origin = "point" -- where the line shoots from (river or otherwise) (x,z)
    layer_number = the number of the layer to edit!
    
    This could be deprecated due to the more general and simpler method
    in removeRedundantVertices (above)
    """
    
    # First, define line segment of interest
    # Know that origin will always be to the right -- more positive
    m = (origin[1] - intersection[1]) / (origin[0] - intersection[0])
    b = origin[1] - m*origin[0]
    
    # Then limit to where values are lower or equal
    self.layer_tops[layer_number] = self.layer_tops[layer_number][self.layer_tops[layer_number][:,1] <= self.layer_tops[layer_number][:,0]*m + b]
    
    
  
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
      

    
    
    
    
