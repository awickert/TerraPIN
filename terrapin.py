#! /user/bin/python

# Started 22 January 2015 by ADW
# based on some earlier notes and code


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
    z_ch = -2
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
