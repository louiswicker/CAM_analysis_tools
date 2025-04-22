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
           "solo": ("/work/wicker/climate_runs/FV3_Solo", "squall_3km"),
           "cm1":  ("/scratch/home/louis.wicker/cm1/run", "squall_3km")
          }

allcape  = ( "C2000", )
allshear = ( "S06", "S12", "S18" )


outprefix = "q_profile"

for run in dirs:

    the_model = run
    the_dir   = os.path.join(dirs[run][0], dirs[run][1])

    print(f"Now computing profiles from this model: {the_model} and from this directory: {the_dir}")

    field = {'w_thres':w_thresh, 'cref_thresh':cref_thresh, 'min_pix': min_pix, 'percentile':percent}

    for shear in allshear:
        for cape in allcape:
        
            label = "%s_%s" % (cape, shear)

            file = os.path.join(the_dir, f"{cape}/{shear}")
            field[label] = generate_ideal_profiles(file, model_type=the_model, w_thresh = w_thresh,
                                                   cref_thresh = cref_thresh, min_pix=min_pix,
                                                   percentile=percent, percent_var='qr', zhgts = zhgts)

    with open(os.path.join(the_dir, f"{cape}/{outprefix}.pkl"), 'wb') as handle:
        pickle.dump(field, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n Squall_profiles wrote file: %s" % os.path.join(the_dir, f"{cape}/{outprefix}.pkl"))
#
