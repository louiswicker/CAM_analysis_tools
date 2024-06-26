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

dirs    = {
           "solo": "/work/wicker/Odin_scr/solo",
           "wrf": "/scratch/wicker/WRF_v4.4.2/test/em_quarter_ss",
           "cm1": "/work/wicker/Odin_scr/cm1r20.3/run",
           "mpas": "/scratch/wicker/MPAS/ideal/squall",
          }

profile_dir = "profiles"

run      = {"solo": "squall_1km", "wrf": "squall_1km", "cm1": "squall_1km"}
run      = {"solo": "squall_1km", "wrf": "squall_1km", "cm1": "squall_1km"}

run      = {"cm1": "squall_3km"}
run      = {"solo": "squall_3km", "wrf": "squall_3km", "cm1": "squall_3km"}

run      = {"wrf": "squall_3km_3rd"}

run      = {"solo": "squall_3km_hdd03"}

run      = {"solo": "squall_3km_n3", "wrf": "squall_3km_dt10", "cm1": "squall_3km_dt10"}
run      = {"solo": "squall_3km_dt30_hdd012"}

run      = {"cm1": "squall_3km_dt10_hdd012", "solo": "squall_3km_dt20_hdd012"}

allcape  = ( "C2000", "C3500")

run      = {"cm1_0": "squall_3km_dt05_hdd012", "cm1_1": "squall_3km_dt05_hdd125", "cm1_2": "squall_3km_dt05_hdd000"}
allcape  = ( "C2000", "C3500" )
allshear = ( "06", "18" )

plabel = "dbz99_4th"

for key in run:

    key2 = key[0:3]

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
