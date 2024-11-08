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

zhgts = 250. + 250.*np.arange(100)

# where is data
dirs    = {
           "mpas": "/work/wicker/climate_runs/MPAS/squall/wofs",
           "wrf": "/work/wicker/climate_runs/WRF/WRF_v4.4.2/ideal/base",
           "cm1": "/work/wicker/climate_runs/cm1r20.3/run/base",
          }

run      = {"wrf": "squall_3km", "mpas": "squall_3km", "cm1": "squall_3km"}

run      = {"mpas": "squall_3km"}

allcape  = ( "C2000", "C2500", "C3000", "C3500")
allshear = ( "06", "12", "18" )

plabel = "precip"

cm1   = {}
wrf   = {}
mpas  = {}
wpas  = {}

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
                cm1[label] = read_cm1_fields(file, file_pattern=None, vars=['w', 'accum_prec'], ret_dbz=True)
                del cm1[label]['w']
                del cm1[label]['dbz']
                
            if key == 'mpas':
                mpas[label] = read_mpas_fields(file, file_pattern=None, vars=['w', 'accum_prec'], ret_dbz=True)
                del mpas[label]['w']
                del mpas[label]['dbz']

            if key == 'wpas':
                wpas[label] = read_mpas_fields(file, file_pattern=None, vars=['w', 'accum_prec'], ret_dbz=True)
                del wpas[label]['w']
                del wpas[label]['dbz']
                
for key in run.keys():
    
    if key == 'solo':
        with open('%s/%s_%s_%s.pkl' % (dirs[key], plabel), 'wb') as handle:
            pickle.dump(solo, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if key == 'wrf':
        with open('%s/%s.pkl' % (dirs[key], plabel), 'wb') as handle:
            pickle.dump(wrf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    if key == 'cm1':
        with open('%s/%s.pkl' % (dirs[key], plabel), 'wb') as handle:
            pickle.dump(cm1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if key == 'mpas':
        with open('%s/%s.pkl' % (dirs[key], plabel), 'wb') as handle:
            pickle.dump(mpas, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if key == 'wpas':
        with open('%s/%s.pkl' % (dirs[key], plabel), 'wb') as handle:
            pickle.dump(wpas, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n Squall_precip wrote pickled file:  %s out!\n" % ('%s/%s.pkl' % (dirs[key], plabel)))
