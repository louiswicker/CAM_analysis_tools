from pathlib import Path
import sys
path = str(Path(Path('File.py').parent.absolute()).parent.absolute())
sys.path.insert(0, path)

import numpy as np
import matplotlib as mlab
import matplotlib.pyplot as plt
import xarray as xr
import os as os
import glob
import matplotlib.gridspec as gridspec
from collections import namedtuple
from analysis_tools import getobjdata

from colormaps import radar_colormap

try:
   import cPickle as pickle
except:
   import pickle

import bz2
import _pickle as cPickle

_nthreads = 2

# where is data
dirs    = {
           "solo": "/work/wicker/climate_runs/FV3_Solo",
           "cm1": "/work/wicker/climate_runs/cm1r20.3/run",
          }

run      = {"solo": "squall", "cm1": "squall"}

allcape  = ( "C2000", )
allshear = ( "06", "12", "18" )

plabel = "object"

for key in run:

    field = {}

    for shear in allshear:
        for cape in allcape:
        
            label = "%s/%s" % (cape, shear)
        
            file = str(os.path.join(dirs[key], "%s/%s" % (run[key], label)))
            field[label] = getobjdata(file, thin_data=4, model_type=key)
                
        
#   with bz2.BZ2File('%s/%s.pbz2' % (dirs[key], plabel), 'w') as handle:
#       cPickle.dump(field, handle)

    with open('%s/%s.pkl' % (dirs[key], plabel), 'wb') as handle:
        pickle.dump(field, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n Squall_obj wrote pickled file:  %s out!\n" % ('%s/%s.pkl' % (dirs[key], plabel)))

# the end
