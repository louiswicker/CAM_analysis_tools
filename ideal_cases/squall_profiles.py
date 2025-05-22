from pathlib import Path
import sys

import numpy as np
import os as os
import glob
import pickle

import timeit

import argparse

import tools

from analysis_tools import generate_ideal_profiles
from tools.thermo import compute_thetae


w_thresh    = 3.0
cref_thresh = 35.
percent     = .99
min_pix     = 3

zhgts = 250. + 250.*np.arange(100)


_cape  = ("QV12", "QV13", "QV14", "QV15", "QV16")
_shear = ( "S06", "S18" )

_cape  = ("QV12",)
_shear = ( "S06", )


_profile_dir = "precip"
_extra_label = ".pkl"

parser = argparse.ArgumentParser()

parser.add_argument('-d', dest="dir", type=str, help="Input directory", default=None)

parser.add_argument('-r', dest="res", type=str, help="Resolution of experiment", default="3km")

parser.add_argument('-t', dest="type", type=str, help="Model core being read", default="solo")

parser.add_argument('--pvar', dest="pvar", type=str, help="Variable for percentiles (qr or dbz)", default="dbz")

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

outprefix = f"WKprofile_{args.pvar}_{int(100*percent)}"

print(outprefix)

the_dir   = os.path.join(args.dir, args.res)

print(f"Now computing profiles from this model: {args.type} and from this directory: {the_dir}")

field = {'w_thres':w_thresh, 'cref_thresh':cref_thresh, 'min_pix': min_pix, 'percentile':percent}

for sh in args.shear:
    for ca in args.cape:

        t1 = timeit.default_timer()
        
        label = "%s/%s" % (ca, sh)

        file = os.path.join(the_dir, label)

        print(f"File now being read:  {file}")
        
        field[label] = generate_ideal_profiles(file, model_type=args.type, w_thresh = w_thresh,
                                               cref_thresh = cref_thresh, min_pix=min_pix,
                                               percentile=percent, percent_var='dbz', zhgts = zhgts)

        t2 = timeit.default_timer()
        
        print(f"Time to read {args.type} {label} file: {t2-t1}")

# with open(os.path.join(the_dir, f"{cape}/{outprefix}.pkl"), 'wb') as handle:
#     pickle.dump(field, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("\n Squall_profiles wrote file: %s" % os.path.join(the_dir, f"{the_dir}/{outprefix}.pkl"))
#
