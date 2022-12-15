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
    
    def write_xr(filename_src, ur,vr,wr, ud,vd,wd, uh,vh,wh, fhour, z, lats, lons):

        ds = xr.Dataset({
                        'ur': xr.DataArray(
            data   = ur, dims = ['fhour', 'nz','ny','nx'], 
            coords={"lats": (["ny","nx"], lats), "lons": (["ny","nx"], lons),"z": (["nz"], z), "fhour": (["fhour"], fhour)} ),
                         'vr': xr.DataArray(
            data   = vr, dims = ['fhour', 'nz','ny','nx'],
            coords={"lats": (["ny","nx"], lats), "lons": (["ny","nx"], lons),"z": (["nz"], z), "fhour": (["fhour"], fhour) } ),
                         'wr': xr.DataArray(
            data   = wr, dims = ['fhour', 'nz','ny','nx'], 
            coords={"lats": (["ny","nx"], lats), "lons": (["ny","nx"], lons),"z": (["nz"], z), "fhour": (["fhour"], fhour) } ), 
                         'ud': xr.DataArray(
            data   = ud, dims = ['fhour', 'nz','ny','nx'],
            coords={"lats": (["ny","nx"], lats), "lons": (["ny","nx"], lons),"z": (["nz"], z), "fhour": (["fhour"], fhour) } ),
                         'vd': xr.DataArray(
            data   = vd, dims = ['fhour', 'nz','ny','nx'],
            coords={"lats": (["ny","nx"], lats), "lons": (["ny","nx"], lons),"z": (["nz"], z), "fhour": (["fhour"], fhour) } ),
                         'wd': xr.DataArray(
            data   = wd, dims = ['fhour', 'nz','ny','nx'], 
            coords={"lats": (["ny","nx"], lats), "lons": (["ny","nx"], lons),"z": (["nz"], z), "fhour": (["fhour"], fhour) } ),  
                          'uh': xr.DataArray(
            data   = uh, dims = ['fhour', 'nz','ny','nx'],
            coords={"lats": (["ny","nx"], lats), "lons": (["ny","nx"], lons),"z": (["nz"], z), "fhour": (["fhour"], fhour) } ),
                         'vh': xr.DataArray(
            data   = vh, dims = ['fhour', 'nz','ny','nx'],
            coords={"lats": (["ny","nx"], lats), "lons": (["ny","nx"], lons),"z": (["nz"], z), "fhour": (["fhour"], fhour) } ),
                         'wh': xr.DataArray(
            data   = wh, dims = ['fhour', 'nz','ny','nx'],
            coords={"lats": (["ny","nx"], lats), "lons": (["ny","nx"], lons),"z": (["nz"], z), "fhour": (["fhour"], fhour) } ) })

        filename = filename_src[:-3]+"_nhhd3D.nc"

        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in ds.data_vars}
        
        ds.to_netcdf(filename, format='NETCDF4', encoding=encoding)
        
        print("Wrote out: ", filename)

    #-----------------------------------------------------------------------------
    # create filenames
        
    hrrr_file = str(os.path.join(in_dir, "%s_%s.nc" % (day, suffix_name[0])))
    rrfs_file = str(os.path.join(in_dir, "%s_%s.nc" % (day, suffix_name[1])))
    
    filenames = (hrrr_file, rrfs_file)
    
    klevels = np.arange(33)
    
    for file in filenames:
        
        tic = time.perf_counter()
        
        ds = xr.open_dataset(file) 

        nhours = ds.dims['fhour']

        print("Opened:  %s, number of forecast hours:  %d" % (file, nhours))

    
        # create NHHD
        
        for t in np.arange(nhours):

            u = np.nan_to_num(ds.u_interp.isel(fhour=t,nz=klevels).values).astype('float64')
            v = np.nan_to_num(ds.v_interp.isel(fhour=t,nz=klevels).values).astype('float64')
            w = np.nan_to_num(ds.w_interp.isel(fhour=t,nz=klevels).values).astype('float64')

            nz, ny, nx = u.shape[0:3]
            
            dims = (nz, ny, nx)  # Z Y, X

            dx   = (0.25, 3, 3)  # dz, dy, dx

            vfield = np.stack((u, v, w), axis=3)

            nhhd = nHHD(grid=dims, spacings=dx)

            nhhd.decompose(vfield)
            
            if t == 0:
                
                ur = np.zeros((nhours, nz, ny, nx))
                vr = np.zeros((nhours, nz, ny, nx))
                wr = np.zeros((nhours, nz, ny, nx))
                ud = np.zeros((nhours, nz, ny, nx))
                vd = np.zeros((nhours, nz, ny, nx))
                wd = np.zeros((nhours, nz, ny, nx))
                uh = np.zeros((nhours, nz, ny, nx))
                vh = np.zeros((nhours, nz, ny, nx))
                wh = np.zeros((nhours, nz, ny, nx))

            ur[t], vr[t], wr[t] = np.squeeze(np.split(nhhd.r, 3, axis=3))

            ud[t], vd[t], wd[t] = np.squeeze(np.split(nhhd.d, 3, axis=3))

            uh[t], vh[t], wh[t] = np.squeeze(np.split(nhhd.h, 3, axis=3))

        write_xr(file, ur,vr,wr, ud,vd,wd, uh,vh,wh, np.arange(nhours), hrrr.z[0:33].values, hrrr.lats.values, hrrr.lons.values)  
        
        toc = time.perf_counter()
        
        print(f"NHHD took {toc - tic:0.4f} seconds")

#--------------- main -------------------

in_dir  = "/work/wicker/ECONUS"
out_dir = "/work/wicker/CAM_analysis_tools/2022_spectra"

#case_days = ["2022050400", 
#             "2022051200",
#             "2022051400",
case_days = ["2022051500",
             "2022051900",
             "2022052300",
             "2022052400",
             "2022052700",
             "2022053000",
             "2022060700"]

for d in case_days:
    print("Running day:  %s\n" % d)
    ret = compute_nhhd(in_dir, out_dir, d)





