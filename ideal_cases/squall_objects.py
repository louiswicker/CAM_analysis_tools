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

#from Plot_tools import *
from pathlib import Path
import sys
path = str(Path(Path('File.py').parent.absolute()).parent.absolute())
sys.path.insert(0, path)

import pickle

_nthreads = 2


# Create data

dirs    = {
           "solo": "/work/wicker/Odin_scr/solo",
           "wrf": "/scratch/wicker/WRF_v4.4.2/test/em_quarter_ss",
           "cm1": "/work/wicker/Odin_scr/cm1r20.3/run",
          }

profile_dir = "object_stat"

run  = {"solo": "squall_3km_n3", "wrf": "squall_3km_dt10", "cm1": "squall_3km_dt10"}
run  = {"solo": "squall_3km_hdd02"}

allcape = ("C2000", "C3500")
allshear = ("06", "18")

for key in run:

    field = {}

    for shear in allshear:
        for cape in allcape:
        
            label = "%s_%s" % (cape, shear)
        
            file = str(os.path.join(dirs[key], "%s_%s" % (run[key], label)))
            field[label] = getobjdata(file, model_type=key)
                
        
    with open('%s/%s_%s_obj.pkl' % (profile_dir, key, run[key]), 'wb') as handle:
        pickle.dump(field, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n Squall_Objects wrote pickled file:  %s out!\n" % ('%s/%s_%s_obj.pkl' % (profile_dir, key, run[key])))

# the end
