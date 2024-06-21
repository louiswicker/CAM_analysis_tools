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

import pickle

_nthreads = 2

# where is data

dirs    = {
           "mpas": "/work/wicker/climate_runs/MPAS/ideal/vis05_3rd",
           "wrf": "/work/wicker/climate_runs/WRF_v4.4.2/ideal/base",
           "cm1": "/work/wicker/climate_runs/cm1r20.3/run/base",
          }

profile_dir = "./climate_runs/object_stat"

allcape = ("C2000", "C3500")
allshear = ("06", "18")

run      = {"cm1": "squall_3km", "mpas": "squall_3km", "wrf": "squall_3km"}
run      = {"mpas": "squall_3km"}

allcape  = ( "C2000", "C3500")
allshear = ( "06", "12", "18" )

plabel = "object"

vars=['hgt', 'w', 'pres', 'pert_th']

for key in run:

    field = {}

    for shear in allshear:
        for cape in allcape:
        
            label = "%s_%s" % (cape, shear)
        
            file = str(os.path.join(dirs[key], "%s_%s" % (run[key], label)))
            field[label] = getobjdata(file, vars=vars, model_type=key)
                
        
    with open('%s/%s.pkl' % (dirs[key], plabel), 'wb') as handle:
        pickle.dump(field, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n Squall_obj wrote pickled file:  %s out!\n" % ('%s/%s.pkl' % (dirs[key], plabel)))

# the end
