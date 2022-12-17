import numpy as np
import xarray as xr
import os as os
import glob
import netCDF4 as ncdf
from pynhhd import nHHD
import time


#---------------------------------------------------------------------------------------------
#
# This code computes the Helmholtz-Hodges decomposition for 3D + time velocity field
#

def compute_nhhd(in_dir, out_dir, day, suffix_name = ["HRRR_ECONUS", "RRFS_ECONUS"]):
    
    def write_xr(filename_src, ur,vr, ud,vd, uh,vh, fhour, z, lats, lons):

        ds = xr.Dataset({
                        'ur': xr.DataArray(
            data   = ur, dims = ['fhour', 'nz','ny','nx'], 
            coords={"lats": (["ny","nx"], lats), "lons": (["ny","nx"], lons),"z": (["nz"], z), "fhour": (["fhour"], fhour)} ),
                         'vr': xr.DataArray(
            data   = vr, dims = ['fhour', 'nz','ny','nx'],
            coords={"lats": (["ny","nx"], lats), "lons": (["ny","nx"], lons),"z": (["nz"], z), "fhour": (["fhour"], fhour) } ),
                         'ud': xr.DataArray(
            data   = ud, dims = ['fhour', 'nz','ny','nx'],
            coords={"lats": (["ny","nx"], lats), "lons": (["ny","nx"], lons),"z": (["nz"], z), "fhour": (["fhour"], fhour) } ),
                         'vd': xr.DataArray(
            data   = vd, dims = ['fhour', 'nz','ny','nx'],
            coords={"lats": (["ny","nx"], lats), "lons": (["ny","nx"], lons),"z": (["nz"], z), "fhour": (["fhour"], fhour) } ),
                          'uh': xr.DataArray(
            data   = uh, dims = ['fhour', 'nz','ny','nx'],
            coords={"lats": (["ny","nx"], lats), "lons": (["ny","nx"], lons),"z": (["nz"], z), "fhour": (["fhour"], fhour) } ),
                         'vh': xr.DataArray(
            data   = vh, dims = ['fhour', 'nz','ny','nx'],
            coords={"lats": (["ny","nx"], lats), "lons": (["ny","nx"], lons),"z": (["nz"], z), "fhour": (["fhour"], fhour) } ) } )

        filename = filename_src[:-3]+"_nhhd2D.nc"

        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in ds.data_vars}
        
        ds.to_netcdf(filename, format='NETCDF4', encoding=encoding)
        
        print("Wrote out: ", filename)

    #-----------------------------------------------------------------------------
    # create filenames
        
    hrrr_file = str(os.path.join(in_dir, "%s_%s.nc" % (day, suffix_name[0])))
    rrfs_file = str(os.path.join(in_dir, "%s_%s.nc" % (day, suffix_name[1])))
    
    filenames = (hrrr_file, rrfs_file)
    
    for file in filenames:
        
        tic = time.perf_counter()
        
        ds = xr.open_dataset(file) 

        nhours = ds.dims['fhour']
    
        nz     = ds.dims['nz']
        nz     = 41

        klevels = np.arange(nz)

        print("Opened:  %s, number of forecast hours:  %d" % (file, nhours))
    
        # create NHHD
        
        for t in np.arange(nhours):

            u = np.nan_to_num(ds.u_interp.isel(fhour=t,nz=klevels).values).astype('float64')
            v = np.nan_to_num(ds.v_interp.isel(fhour=t,nz=klevels).values).astype('float64')

            ny, nx = u.shape[1:3]

            dims = (ny, nx)  # Z Y, X

            dx   = (3, 3)  # dz, dy, dx

            if t == 0:
                
                ur = np.zeros((nhours, nz, ny, nx))
                vr = np.zeros((nhours, nz, ny, nx))
                ud = np.zeros((nhours, nz, ny, nx))
                vd = np.zeros((nhours, nz, ny, nx))
                uh = np.zeros((nhours, nz, ny, nx))
                vh = np.zeros((nhours, nz, ny, nx))

            for k in klevels:

                vfield = np.stack((u[k], v[k]), axis=2)

                nhhd = nHHD(grid=dims, spacings=dx)

                nhhd.decompose(vfield)


                ur[t,k], vr[t,k] = np.squeeze(np.split(nhhd.r, 2, axis=2))

                ud[t,k], vd[t,k] = np.squeeze(np.split(nhhd.d, 2, axis=2))
    
                uh[t,k], vh[t,k] = np.squeeze(np.split(nhhd.h, 2, axis=2))

        write_xr(file, ur,vr, ud,vd, uh,vh, np.arange(nhours), ds.z[0:nz].values, ds.lats.values, ds.lons.values)  
        
        toc = time.perf_counter()
        
        print(f"NHHD took {toc - tic:0.4f} seconds")

#--------------- main -------------------

in_dir  = "/work/wicker/ECONUS"
out_dir = "/work/wicker/CAM_analysis_tools/2022_spectra"

case_days = ["2022050400",
             "2022051200",
             "2022051400",
             "2022051500",
             "2022051900",
             "2022052300",
             "2022052400",
             "2022052700",
             "2022053000",
             "2022060700"]

for d in case_days:
    print("Running day:  %s\n" % d)
    ret = compute_nhhd(in_dir, out_dir, d)
