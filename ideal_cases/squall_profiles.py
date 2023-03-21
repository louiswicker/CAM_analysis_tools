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


w_thresh = 5.0
cref_thresh = 35.
percent = None

zhgts = 250. + 250.*np.arange(100)

dirs    = {
           "solo": "/work/wicker/Odin_scr/solo",
           "wrf": "/work/wicker/WRF/WRF/test/em_quarter_ss",
           "cm1": "/work/wicker/Odin_scr/cm1r20.3/run",
          }

run      = {"solo": "squall_3km", "solo": "squall_nw0", "wrf": "squall_3km", "cm1": "squall_3km"}
run      = {"solo": "squall_3km"}
allcape  = ("C2000", "C3500")
allshear = ("06", "12", "18")

solo  = {}
cm1   = {}
wrf   = {}

for key in run:

    field = {}

    for shear in allshear:
        for cape in allcape:
        
            label = "%s_%s" % (cape, shear)

            file = str(os.path.join(dirs[key], "%s_%s" % (run[key], label)))
            field[label] = generate_ideal_profiles(file, model_type=key, w_thresh = w_thresh,
                                                   cref_thresh = cref_thresh,
                                                   percentile=percent, zhgts = zhgts)

    with open('%s_%s_35dbz_profiles.pkl' % (key, run[key]), 'wb') as handle:
        pickle.dump(field, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n Compute_Profiles wrote pickled file:  %s out!\n" % ('%s_%s_profiles.pkl' % (key, run[key])))
