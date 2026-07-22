from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages
from setuptools.command.install import install

import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "Terrapin",
    version = "0.0.0",
    packages = find_packages(exclude="tests"),

    install_requires = [
        "numpy>=2",         # the geometry engine targets NumPy 2
        "shapely>=2",       # GEOS predicates / polygon algebra
        "scipy",            # root-finders for the colluvial-pile solver
        "matplotlib",       # cross-section plotting
    ],
    #entry_points = {
    #  'console_scripts': ['terrapin = terrapin:main']
    #  },

    package_data = { 
      '': ['*.md']
      },

    # metadata for upload to PyPI
    author = "Andrew D. Wickert",
    author_email = "awickert@umn.edu",
    description = "River terrace development",
    license = "GPL v3",
    keywords = "fluvial geomorphology Quaternary river",
    url = "http://csdms.colorado.edu/wiki/Model:Terrapin",
    download_url = "https://github.com/awickert/Terrapin",
    long_description=read('README.md'),
)
