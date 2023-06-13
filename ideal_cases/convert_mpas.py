import numpy as np
import xarray as xr
import os as os
from scipy.spatial import KDTree
import argparse

_in_filename  = "output.nc"

dx = 3000.
nx = 254
ny = 254

output_variables = {'w':'w', 'u':'uReconstructZonal', 'v':'uReconstructMeridional', 'accum_precip':'rainnc', \
                    'theta':'theta', 'theta_base':'theta_base', 'den':'rho', 
                    'press':'pressure', 'qv':'qv', 'qc':'qc', 'qr':'qr'}

parser = argparse.ArgumentParser()
parser.add_argument('-i', dest="infile", type=str, 
                    help="Filename for raw output from MPAS on hex grid", default=_in_filename)
parser.add_argument('-o', dest="outfile", type=str, \
                    help="Filename for interpolated out from MPAS on quad grid",default=None)

args = parser.parse_args()

infile = args.infile

if args.outfile:
    outfile = args.outfile
else:
    outfile = ("%s_quad.nc") % infile[0:-3]


# Open input file, read cell centered coordinates

f = xr.open_dataset(infile)
xC = f.xCell.values
yC = f.yCell.values
zE = f.zgrid.values[0,:]
zC = 0.5*(zE[1:] + zE[:-1])

print(xC.max(), xC.min())
print(yC.max(), yC.min())
print(zC.max(), zC.min())

nlevels = f.nVertLevels.shape[0]
ntimes = f.Time.shape[0]

print(nx)
print(ny)
print(ntimes)
print(nlevels)

# Create new grid for interpolation, create KDTree mapping

xg = dx*(0.5+np.arange(nx))
yg = dx*(0.5+np.arange(ny))
xx, yy = np.meshgrid(xg,yg)

coord_C = list(zip(xC, yC))

coord_G = list(zip(xx.flatten(), yy.flatten()))

tree = KDTree(coord_C)
dis, index = tree.query(coord_G, k=1)

interp_arrays = {}

for key in output_variables:
    
    fldC = f[output_variables[key]].values

    if len(fldC.shape) == 2:

        fld_interp = fldC[:,index].reshape(ntimes, ny, nx)
        
        interp_arrays[key] = [len(fldC.shape), fld_interp, ntimes, ny, nx]
    
    elif len(fldC.shape) == 3:

        fldT = np.moveaxis(fldC, -1, 1)
        
        nz = fldT.shape[1]
        
        if nz > nlevels:     # interp w to zone centers
            fldT = 0.5 * (fldT[:,1:,:] + fldT[:,:-1,:])
        
        fld_interp = fldT[:,:,index].reshape(ntimes,nlevels,ny,nx)
        
        interp_arrays[key] = [len(fldT.shape), fld_interp, ntimes, nlevels, ny, nx]
        
    else:
        print("%s variable is not yet implemented, dimensions are wrong - DIMS:  %i3.3" \
               % (output_variables[key], len(fldC.shape)))

# Write to XARRAY file (netCDF4)

for n, key in enumerate(interp_arrays):
        
    if interp_arrays[key][0] == 2:  # 2D spatial data set like precip
        
        new = xr.DataArray( interp_arrays[key][1], dims=['nt', 'ny', 'nx'],
                            coords={"time": (["nt"], np.arange(ntimes)),
                                    "x": (["nx"], xg),
                                    "y": (["ny"], yg) } )
    else:
        
        new = xr.DataArray( interp_arrays[key][1], dims = ['nt', 'nz', 'ny', 'nx'],
                            coords={"time": (["nt"], np.arange(ntimes)),
                                    "x": (["nx"], xg),
                                    "y": (["ny"], yg),
                                    "z": (["nz"], zC) } )
        
    if n == 0:

        ds_conus = new.to_dataset(name = key)

    else:

        ds_conus[key] = new
        
    print("Wrote %s" % key)

ds_conus.to_netcdf(outfile, mode='w')
print(f'Successfully wrote interpolated MPAS data to file:: {outfile}','\n')
