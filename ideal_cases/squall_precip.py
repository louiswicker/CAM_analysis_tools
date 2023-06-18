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
import pickle

import tools

from analysis_tools import read_solo_fields, read_wrf_fields, read_cm1_fields, read_mpas_fields

w_thresh = 5.0
cref_thresh = 35.
percent = None

zhgts = 250. + 250.*np.arange(100)

dirs    = {
           "solo": "/work/wicker/Odin_scr/solo",
           "wrf": "/work/wicker/WRF/WRF/test/em_quarter_ss",
           "cm1": "/work/wicker/Odin_scr/cm1r20.3/run",
           "mpas": "/scratch/wicker/MPAS/ideal/squall",
          }

profile_dir = "profiles"

run      = {"solo": "squall_3km", "wrf": "squall_3km", "cm1": "squall_3km", "mpas": "squall_3km"}
run      = {"wrf": "squall_3km_3rd"}
allcape  = ("C2000", "C3500")
allshear = ( "06", "18" )

solo  = {}
cm1   = {}
wrf   = {}
mpas  = {}

for key in run:

    field = {}

    for shear in allshear:
        for cape in allcape:
        
            label = "%s_%s" % (cape, shear)


            file = str(os.path.join(dirs[key], "%s_%s" % (run[key], label)))

            if key == 'solo':
                solo[label] = read_solo_fields(file, file_pattern=None, vars=['accum_prec'])
                
            if key == 'wrf':

                wrf[label] = read_wrf_fields(file, file_pattern=None, vars=['accum_prec'])

            if key == 'cm1':
                cm1[label] = read_cm1_fields(file, file_pattern=None, vars=['accum_prec'])
                
            if key == 'mpas':
                mpas[label] = read_mpas_fields(file, file_pattern=None, vars=['accum_prec'])
                
for key in run.keys():
    
    if key == 'solo':
        with open('precip/solo_%s_accum_prec.pkl' % run['solo'], 'wb') as handle:
            pickle.dump(solo, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if key == 'wrf':
        with open('precip/wrf_%s_accum_prec.pkl' % run['wrf'], 'wb') as handle:
            pickle.dump(wrf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    if key == 'cm1':
        with open('precip/cm1_%s_accum_prec.pkl' % run['cm1'], 'wb') as handle:
            pickle.dump(cm1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if key == 'mpas':
        with open('precip/mpas_%s_accum_prec.pkl' % run['mpas'], 'wb') as handle:
            pickle.dump(mpas, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n Squall_precip wrote pickled file:  %s out!\n" % \
            ('%s/%s_%s_35dbz_profiles.pkl' % (profile_dir, key, run[key])))

