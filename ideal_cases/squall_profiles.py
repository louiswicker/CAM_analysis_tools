from pathlib import Path
import sys

import numpy as np
import matplotlib as mlab
import matplotlib.pyplot as plt
import xarray as xr
import os as os
import glob
import matplotlib.gridspec as gridspec
import metpy.calc as mpcalc
from metpy.units import units
import pickle

import tools

from analysis_tools import generate_ideal_profiles
from tools.thermo import compute_thetae


w_thresh = 3.0
cref_thresh = 35.
percent = .99
min_pix = 3

zhgts = 250. + 250.*np.arange(100)

# where is data

dirs    = {
           "mpas": "/work/wicker/climate_runs/MPAS/ideal/vis01_3rd",
           "wrf": "/work/wicker/climate_runs/WRF_v4.4.2/ideal/thetaM",
           "cm1": "/work/wicker/climate_runs/cm1r20.3/run/base",
          }

profile_dir = "./climate_runs/profiles"

allcape = ("C2000", "C3500")
allshear = ("06", "18")

run      = {"cm1": "squall_3km", "mpas": "squall_3km", "wrf": "squall_3km"}

run      = {"mpas": "squall_3km", "wrf": "squall_3km"}
run      = {"mpas": "squall_3km" }

allcape  = ( "C2000", "C3500")
allshear = ( "06", "12", "18" )

plabel = "3rd"

for key in run:

    key2 = key[0:4]

    field = {'w_thres':w_thresh, 'cref_thresh':cref_thresh, 'min_pix': min_pix, 'percentile':percent}

    for shear in allshear:
        for cape in allcape:
        
            label = "%s_%s" % (cape, shear)

            file = str(os.path.join(dirs[key2], "%s_%s" % (run[key], label)))
            field[label] = generate_ideal_profiles(file, model_type=key2, w_thresh = w_thresh,
                                                   cref_thresh = cref_thresh, min_pix=min_pix,
                                                   percentile=percent, zhgts = zhgts)

    with open('%s/%s_%s_%s.pkl' % (profile_dir, key2, run[key], plabel), 'wb') as handle:
        pickle.dump(field, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n Squall_profiles wrote pickled file:  %s out!\n" % ('%s/%s_%s_%s.pkl' % (profile_dir, key, run[key], plabel)))
