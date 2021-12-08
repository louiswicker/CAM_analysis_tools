#!/usr/bin/env python
#
#
import os, sys
import glob
from optparse import OptionParser

import numpy as N
import pylab as P
import netCDF4 as netcdf
from time import clock
import raymond_lowpass

#---------------------------------------------------------------------------------------------------
# Main function defined to return correct sys.exit() calls
#
def main(argv=None):
    if argv is None:
           argv = sys.argv

    # Currently configured to take grid spacing as argument, not eps; assumes eps has already been tuned for specified grid spacing (need to modify below)
    # Usage: python filter.py -f ./wrfout_d01_0001-01-01_01:00:00 --cutoff 2000
  
    parser = OptionParser()
    parser.add_option("-f", "--file",   dest="file",    type="string", help="Raw file to be filtered")
    parser.add_option("-v",             dest="var",     type="string", help="Variable to be filtered (else all vars filtered)")
    parser.add_option("--dx1",          dest="dx1",     type="float",  help="grid spacing (m)")
    parser.add_option("--cutoff",       dest="cutoff",  type="float",  help="desired cutoff wavelength of filtered fields (m)")

    (options, args) = parser.parse_args()

    if options.file:
        print "\nInput netCDF file:  ",   options.file
        file = options.file
    else:
        print "\nNo input netCDF file exiting" 
        sys.exit(1)

    if options.dx1 == None:
        dx1 = 1000/3.
    else:
        dx1 = float(options.dx1)

    # Need to modify!!!

    if (options.cutoff==2000): 
      eps = 0.3e2
    elif (options.cutoff==4000): 
      eps = 2e3
    elif (options.cutoff==8000): 
      eps = 1.5e5
    elif (options.cutoff==16000): 
      eps = 6.5e6
    else:
      print "no eps value for this cutoff!"
      sys.exit(1)

    # Original wrfout/wrfrst file is retained; filtered fields are put into new file

    file_new = file + "_%dkm" % (options.cutoff/1000)

    cmd = "cp %s %s" % (file, file_new)
    print cmd
    os.system(cmd)
   
    ncfile  = netcdf.Dataset(file, 'r')
    ncfile2 = netcdf.Dataset(file_new, 'r+') 

    if options.var == None:
      variable_list = ncfile.variables
    else:
      variable_list = [options.var]

    cpu0 = clock()
    for item in variable_list:

      for ti in range(0, len(ncfile.dimensions['Time'])):
        print item
        cpu1 = clock()

        farray = ncfile.variables[item][ti] 
        print ncfile.variables[item].dimensions[:]

        if not ((ncfile.variables[item].dimensions[-1] == 'west_east_stag' or ncfile.variables[item].dimensions[-1] == 'west_east') and
            (ncfile.variables[item].dimensions[-2] == 'south_north_stag' or ncfile.variables[item].dimensions[-2] == 'south_north')):

          print "Copying (not filtering) %s" % (item)  # if field does not contain both horizontal dimensions, obviously not going to filter
          ncfile2.variables[item][:] = ncfile.variables[item][:]
          continue

        elif (len(ncfile.variables[item].dimensions[:]) == 3):  # 2D fields
          ncfile2.variables[item][:] = ncfile.variables[item][:]
          filtered = raymond_lowpass.raymond2d_lowpass(farray.transpose(),eps) # actual application of the filter
          #print ti, ncfile2.variables[item][0,:,:].shape, filtered.shape
          ncfile2.variables[item][ti] = filtered.transpose()

        else: # 3D fields (loop through vertical levels)

          for k in range(0, farray.shape[0]):
              filtered = raymond_lowpass.raymond2d_lowpass(farray[k].transpose(),eps)
              if (k==0): print ti, ncfile2.variables[item][0,k,:,:].shape, filtered.shape
              ncfile2.variables[item][ti,k] = filtered.transpose()
        cpu1 = clock() - cpu1
        print "\nTotal time to filter the variable is:  %f \n" % (cpu1)


    cpu0 = clock() - cpu0
    print "\nTotal time to filter all fields is:  %f \n" % (cpu0)

    ncfile.close()
    ncfile2.close()

#-------------------------------------------------------------------------------
# Main program for testing...
#
if __name__ == "__main__":
    sys.exit(main())
    
# End of file
