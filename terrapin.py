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

  def initialize(self):
    self.set_input_values()

  def update(self):
    pass

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
    z_ch = 0
    # elevation of the channel alluvial surface at the starting time
    eta = 0
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
    K_r = 1E-4
    # Surface elevation profiles -- start out flat
    # bedrock (x,z)
    z_br = np.array([[-np.inf, 0], [0, 0]])
    # sediment (x,z) -- actually, this would be surface elevation
    # Does sediment just exist in a little area or across the entire thing?
    # Must exist above bedrock.
    # So all units must be in order from bottom to top.
    # But none exists now! Hm, have to think.
    eta = np.array([[-np.inf, 0], [0, 0]])
    
    # Create arrays of values of angles and resistance to erosion
    # Alluvium is always the last one
    # Goes from bottom of strat column to top
    alpha = [alpha_r, alpha_a]
    k = [k_r, k_a]
    lith = [z_br, eta]
    
    # Intersection-finding
    
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
    pass

  def aggrade(self):
    """
    Fill up area within the valley with sediment (or disperse sediment 
    over some distance on a flat plain? Or just error/quit in that case?
    """
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
    
    
    
    
  # Utility functions
  def evaluatePiecewiseLinear(self, x, pwl)  
    """
    Evaluates a piecewise linear expression to solve for z at a given x.
    x:   the x-value
    pwl: a 2-column numpy array, ([x, z]), that must contain at least
         four entries (two points define a line)
    """
    
    # First, define line segment of interest
    # No internal error handling -- will just let it crash if the point
    # is outside the reach of the piecewise linear segment
    xmin_pwl = np.max( pwl[:,0] <= x )
    xmax_pwl = np.min( pwl[:,0] >= x )
    z_xmin_pwl = pwl[:,1][pwl[:,0] == xmin_pwl]
    z_xmax_pwl = pwl[:,1][pwl[:,0] == xmax_pwl]
    
    # In case the point is at intersection of two segments, just give it 
    # the known z-value
    # (this could have been avoided by using < and >= instead of <= and >=,
    # but then that could cause problems of the point to be selected were 
    # on the edge of the domain but still hanging on to the last defined 
    # point
    if xlim_pwl == xmax_pwl:
      # In this case, z_xmin_pwl and z_xmax_pwl are defined to be the same,
      # so just pick one
      z = z_xmin_pwl
    else
      # define the line given by the segment endpoints
      # z = mx + b 
      # slope
      m = (z_xmax_pwl - z_xmin_pwl)/(xmax_pwl - xmin_pwl)
      # intercept -- could equally calculate with x_max
      b = z_xmin_pwl - m*xmin_pwl
      # Now can compute z
      z = m*x + b
      
    return z

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
