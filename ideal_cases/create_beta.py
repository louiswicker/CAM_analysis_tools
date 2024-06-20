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
import matplotlib.gridspec as gridspec
from collections import namedtuple
from analysis_tools import getobjdata

from colormaps import radar_colormap

from tools.File_tools import write_forcing

from tools.FV3_tools import read_solo_fields
from tools.CM1_tools import read_cm1_fields

_retrievebeta = "./retrievebeta.exe"

# Input directories

dirs    = {
       #   "solo": "/work/wicker/Odin_scr/solo",
           "cm1": "/work/wicker/Odin_scr/cm1r20.3/run",
          }

out_dir = "beta_diagnostic"

allcape = ("C2000", "C3500")
allshear = ("06", "18")

run      = {"cm1": "bubble_3km_TESTA_hdd125", "solo": "bubble_3km_TESTA_hdd125"}
run      = {"cm1": "bubble_3km_DENSEOUTPUT_hdd125" }

allcape  = ( "C2000", )
allshear = ( "00", )

for key in run:

    for shear in allshear:
        for cape in allcape:
        
            label = "%s_%s" % (cape, shear)
        
            infile = str(os.path.join(dirs[key], "%s_%s" % (run[key], label)))
            oufile = str(os.path.join(out_dir, "%s_%s_%s_beta_in.nc" % (key, run[key], label)))
            befile = str(os.path.join(out_dir, "%s_%s_%s_beta_out.nc" % (key, run[key], label)))

            print("\n Reading file:  %s" % infile)

            if key == 'cm1':

                ds = read_cm1_fields(infile, vars = ['den'], \
                                     file_pattern=None, ret_dbz=False, ret_ds=False)

                write_forcing(ds, oufile)
        
            if key == 'solo':

                ds = read_solo_fields(infile, vars = ['den'], \
                                     file_pattern=None, ret_dbz=False, ret_ds=False)

                write_forcing(ds, oufile)

            print("\n Wrote out beta forcing file:  %s" % oufile)

            cmd = "%s -i ./%s -o ./%s" % (_retrievebeta, oufile, befile)
            print(" \n Running cmd: %s" % cmd)

            ret = os.system(cmd)

            if ret == 0:
                print("\n Beta solution file  %s succesfully write" % befile)
            else:
                print("\n Beta solution did not execute....")
                sys.exit(1)
# the end
