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

import timeit

from analysis_tools import read_solo_fields, read_wrf_fields, read_cm1_fields, read_mpas_fields

import argparse

_cape  = ("QV12", "QV13", "QV14", "QV15", "QV16")
_shear = ( "S06", "S18" )

_profile_dir = "precip"
_extra_label = ".pkl"

parser = argparse.ArgumentParser()

parser.add_argument('-d', dest="dir", type=str, help="Input directory", default=None)

parser.add_argument('-r', dest="res", type=str, help="Resolution of experiment", default="3km")

parser.add_argument('-t', dest="type", type=str, help="model core being read", default="solo")

parser.add_argument('--cape', dest="cape", default=_cape)

parser.add_argument('--shear', dest="shear", default=_shear)

args = parser.parse_args()

if args.dir == None:
    print("Need to provide directory path\n")
    sys.exit()
else:
    exp_name = os.path.basename(args.dir)

if args.res == None:
    print("Need to provide resolution (3km, 1km, etc)\n")
    sys.exit()

model = {}

for sh in args.shear:
    for ca in args.cape:

        t1 = timeit.default_timer()
    
        label = "%s/%s" % (ca, sh)

        file = str(os.path.join(args.dir, "%s/%s" % (args.res, label)))

        if args.type == 'solo':
            model[label] = read_solo_fields(file, file_pattern=None, vars=['accum_prec'])

        if args.type == 'wrf':

            model[label] = read_wrf_fields(file, file_pattern=None, vars=['accum_prec'])

        if args.type == 'cm1':
            model[label] = read_cm1_fields(file, file_pattern=None, vars=['accum_prec'])
            
        if args.type == 'mpas':
            model[label] = read_mpas_fields(file, file_pattern=None, vars=['accum_prec'])

        t2 = timeit.default_timer()

        print(f"Time to read {args.type} {label} file: {t2-t1}")
                
with open('%s/%s_%s%s' % (_profile_dir, exp_name, args.res, _extra_label), 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("\n Squall_precip wrote pickled file:  %s out!\n" % ('%s/%s_%s%s' % (_profile_dir, exp_name, args.res, _extra_label)))
