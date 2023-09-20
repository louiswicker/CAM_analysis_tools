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
from tools.FV3_tools import read_solo_fields

_nthreads = 2

#---------------------------------------------------------------------
# local interp function

def interp3dz_np(data, z3d, z1d, nthreads = _nthreads):

    dinterp = np.zeros(len(z1d),dtype=np.float32)

    dinterp[:] = np.interp(z1d, z3d[:], data[:])

    return dinterp
#---------------------------------------------------------------------

w_thresh = 1.0
cref_thresh = -10.
percent = None
min_pix = 1

zhgts = 250. + 250.*np.arange(100)

dirs    = {
           "solo": "/work/wicker/Odin_scr/solo",
           "wrf": "/work/wicker/WRF/WRF/test/em_quarter_ss",
           "cm1": "/work/wicker/Odin_scr/cm1r20.3/run",
           "mpas": "/scratch/wicker/MPAS/ideal/squall",
          }

profile_dir = "profiles"

run      = {"solo": "squall_1km", "wrf": "squall_1km", "cm1": "squall_1km"}
run      = {"solo": "squall_1km", "wrf": "squall_1km", "cm1": "squall_1km"}

run      = {"cm1": "squall_3km"}
run      = {"solo": "squall_3km", "wrf": "squall_3km", "cm1": "squall_3km"}

run      = {"wrf": "squall_3km_3rd"}

run      = {"solo": "squall_3km", "wrf": "squall_3km", "cm1": "squall_3km", "mpas": "squall_3km"}
allcape  = ("C2000", "C3500")

run      = {"solo": "squall_3km_bubble2_d412", "solo1": "squall_3km_bubble2_d406", "solo2":"squall_3km_bubble2_d403"}

plabel = "3km_bubble2"

field = {'w_thres':w_thresh, 'cref_thresh':cref_thresh, 'min_pix': min_pix, 'percentile':percent}

for key in run:

    key2 = key[0:4]

    label = "%s" % (run[key].split('_')[-1])

    print(key2, run[key].split('_'), label)

    file = str(os.path.join(dirs[key2], "%s" % (run[key])))
#   field = generate_ideal_profiles(file, model_type=key2, w_thresh = w_thresh,
#                                                  cref_thresh = cref_thresh, min_pix=min_pix,
#                                                  percentile=percent, zhgts = zhgts)

#---------------------------
    if key2 == 'solo':

        ds = read_solo_fields(file, vars=['hgt', 'pres', 'w', 'temp', 'buoy', 'theta', 'pert_t', 'pert_th',
                                    'qv', 'pert_p' ], ret_dbz=True)

        ds['thetae'] = compute_thetae(ds)

    nt, nz, ny, nx = ds['w'].shape
    print(nt, nz, nx, ny)

#   field['w']  = interp3dz_np(ds['w'][:,:,ny//2,nx//2].transpose(),   \
#                              ds['hgt'][:,:,ny//2,nx//2].transpose(), \
#                              zhgts, nthreads = _nthreads)
#

    fld = {}
    w   = []

    for n in np.arange(nt):

        ind = np.unravel_index(np.argmax(ds['w'][n].max(axis=0)), (ny,nx))

        iloc = [ind[1]-1,ind[1],ind[1]+1]
        jloc = [ind[0]-1,ind[0],ind[0]+1]
        wfld = ds['w'][n,:, jloc, iloc].mean(0)
        dhgt = ds['hgt'][n,:,jloc,iloc].mean(0)
        w.append(interp3dz_np(wfld, dhgt, zhgts, nthreads = _nthreads))

    field['w'] = np.asarray(w, dtype=np.float32).transpose()

    with open('%s/%s_%s.pkl' % (profile_dir, key, plabel), 'wb') as handle:
        pickle.dump(field, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n Squall_profiles wrote pickled file:  %s out!\n" % ('%s/%s_%s.pkl' % (profile_dir, key, plabel)))
