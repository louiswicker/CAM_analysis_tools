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

from tools.File_tools import write_forcing

from tools.FV3_tools import read_solo_fields
from tools.CM1_tools import read_cm1_fields

_retrievebeta = "/home/louis.wicker/p_decomp/run/retrievebeta.exe"

zlevels = 10.0 + 100.*np.arange(200)

# Input directories

dirs    = {
           "solo": "/work/wicker/climate_runs/FV3_Solo/bubble/euler",
           # "cm1":  "/work/wicker/climate_runs/cm1r20.3/run/bubble",
          }

exp     = { "cm1": "3km", "solo": "3km" }
exp     = { "solo": "3km" }

allcape = ("QV13", "QV16")
allcape = ( "QV13", )
allshear = ( "S00", )

for key in exp:

   for shear in allshear:
        for cape in allcape:
        
            label = f"{cape}" # _{shear}
        
            infile = str(os.path.join(dirs[key], "%s/%s" % (exp[key], label)))
            oufile = str(os.path.join(dirs[key], "%s/%s/total_den.nc" % (exp[key], label)))
            befile = str(os.path.join(dirs[key], "%s/%s/w_b_accel.nc" % (exp[key], label)))

            print(f"Reading history file:        {infile}")
            print(f"Name of output density file: {oufile}")
            print(f"Name of output beta    file: {befile}")

            if key == 'cm1':

                ds = read_cm1_fields(infile, vars = ['rho',], zinterp=zlevels )

                write_forcing(ds, oufile)
        
            if key == 'solo':

                ds = read_solo_fields(infile, vars = ['rho'], zinterp=zlevels )

                write_forcing(ds, oufile)

            print("\n Wrote out beta forcing file:  %s" % oufile)

            cmd = "%s -i %s -o %s" % (_retrievebeta, oufile, befile)
            print(" \n Running cmd: %s" % cmd)

            ret = os.system(cmd)

            if ret == 0:
                print("\n Beta solution file  %s succesfully write" % befile)
            else:
                print("\n Beta solution did not execute....")
                sys.exit(1)
# the end
