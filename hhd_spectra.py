import numpy as np
import matplotlib as mlab
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import xarray as xr
import os as os
import matplotlib.gridspec as gridspec
import glob
import netCDF4 as ncdf
import time

# Local import 

from real_cases.spectra.py_spectra import *

#-------------------------------------------------------

analysis_levels = [4, 16, 28, 38]

in_dir  = "/work/wicker/ECONUS"
out_dir = "/work/wicker/CAM_analysis_tools/2022_new_spectra"

suffix_names = ["HRRR","RRFS"]
region       = "ECONUS"

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

func   = get_spectra2D_RAD
dtrend = True

#-------------------------------------------------------

def update_ticks(x, pos):
    if x != 0.0:
        return "%2.1f" % (2.0/x)
    else:
        return r'$\infty$'

#-------------------------------------------------------
# main code...

ncases = len(case_days)

avg_grid = 0.005 + 0.985* np.linspace(0.0, 1.0, num=100, endpoint=True)

ret_rke =  {"HRRR":[], 
            "RRFS":[]}

ret_dke =  {"HRRR":[], 
            "RRFS":[]}

ret_tke =  {"HRRR":[], 
            "RRFS":[]}

avg_rke = {"HRRR":np.zeros((ncases,avg_grid.shape[0],)), 
           "RRFS":np.zeros((ncases,avg_grid.shape[0],))}
avg_dke = {"HRRR":np.zeros((ncases,avg_grid.shape[0],)), 
           "RRFS":np.zeros((ncases,avg_grid.shape[0],))}
avg_tke = {"HRRR":np.zeros((ncases,avg_grid.shape[0],)), 
           "RRFS":np.zeros((ncases,avg_grid.shape[0],))}

#

print("\n====> Begin processing runs\n")

for k in analysis_levels:
                
    # these levels are averaged together

    klevels = [k-1, k, k+1]

    for n, day in enumerate(case_days):
        
        for run in suffix_names:

            print("\n----> Processing run: %s for day:  %s \n" % (run, day))
            
            file1  = str(os.path.join(in_dir, "%s_%s_%s_hhd.nc" % (day, run, region)))
            print("          %s\n" % file1)
            
            file2  = str(os.path.join(in_dir, "%s_%s_%s.nc" % (day, run, region)))
            print("          %s\n" % file2)
            
            # Open data set

            ds = xr.open_dataset(file1)
            do = xr.open_dataset(file2)
            
            zlevels = ds.z
            
            nhours = ds.dims['fhour']

            store_rke = []
            store_dke = []
            store_tke = []
            
            for t in np.arange(nhours):
                
                print("          F-HOUR:  %d  K-LEVEL:  %d" % (t, k))
            
                # Convert to numpy arrays, fill in zeros, compute horizontal TKE.

                u  = np.nan_to_num(do.u_interp.isel(fhour=t,nz=klevels).values).astype('float64')
                v  = np.nan_to_num(do.v_interp.isel(fhour=t,nz=klevels).values).astype('float64')

                ur = np.nan_to_num(ds.ur.isel(fhour=t,nz=klevels).values).astype('float64')
                vr = np.nan_to_num(ds.vr.isel(fhour=t,nz=klevels).values).astype('float64')

                ud = np.nan_to_num(ds.ud.isel(fhour=t,nz=klevels).values).astype('float64')
                vd = np.nan_to_num(ds.vd.isel(fhour=t,nz=klevels).values).astype('float64')

                tke = 0.5*(u**2  + v**2)
                rke = 0.5*(ur**2 + vr**2)
                dke = 0.5*(ud**2 + vd**2)

                # Compute rke spectra...

                ret = get_spectraND(rke, func = func, dtrend = dtrend)

                store_rke.append(np.interp(avg_grid, ret[2], ret[1]))

                # Compute dke spectra...

                ret = get_spectraND(dke, func = func, dtrend = dtrend)

                store_dke.append(np.interp(avg_grid, ret[2], ret[1]))

                # Compute hke spectra...

                ret = get_spectraND(tke, func = func, dtrend = dtrend)

                store_tke.append(np.interp(avg_grid, ret[2], ret[1]))
            
            # Average spectra over all the fhours
            
            avg_rke[run][n][:] = np.asarray(store_rke).mean(axis=0)
            avg_dke[run][n][:] = np.asarray(store_dke).mean(axis=0)
            avg_tke[run][n][:] = np.asarray(store_tke).mean(axis=0)

            ds.close()
            do.close()
            
    # Now average spectra over all the days...

    rke_hrrr = avg_rke['HRRR'].mean(axis=0)
    rke_rrfs = avg_rke['RRFS'].mean(axis=0)
    
    dke_hrrr = avg_dke['HRRR'].mean(axis=0)
    dke_rrfs = avg_dke['RRFS'].mean(axis=0)

    tke_hrrr = avg_tke['HRRR'].mean(axis=0)
    tke_rrfs = avg_tke['RRFS'].mean(axis=0)

    # Now plot the average spectra

    fig = plt.figure(constrained_layout=True,figsize=(20,10))

    gs = gridspec.GridSpec(1, 2, figure=fig)

    axes = fig.add_subplot(gs[0, 0])

    legend='HRRR:black\n\nRRFS:red\n\nRotation=solid\nDivergent=dashed'

    axes.loglog(avg_grid, rke_hrrr, color='black', linewidth=2.)
    axes.loglog(avg_grid, rke_rrfs, color='red',   linewidth=2.)
    
    axes.loglog(avg_grid, dke_hrrr, color='black', linestyle='--')
    axes.loglog(avg_grid, dke_rrfs, color='red',   linestyle='--')

    axes.set_xlim(2.0/avg_grid.shape[0], 1.0)

    axes.annotate("%s" % legend, xy=(0.10, 0.15), xycoords='axes fraction', color='k',fontsize=18)
    axes.xaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))

    if k < 8:
            ylim = [1.0e10,1.0e18]
    else:
        ylim = [1.0e12,1.0e20]
        
    axes.set_ylim(ylim[0], ylim[1])

    ylabel = 5. * ylim[0]

    xoffset = [0.01, 0.005, 0.0035, 0.001]

    for n, w in enumerate([4.0, 8.0, 12.0, 16.0]):
        axes.axvline(x = (2.0/w), color = 'grey', label = 'axvline - full height')  
        axes.annotate(r"%d$\Delta$x" % w, xy=(2.0/w + xoffset[n], ylabel), xycoords='data', color='k',fontsize=12)

    axes.set_xlabel(r"Wavelength in ($\Delta$x)", fontsize=16)
    axes.set_ylabel(r"E(k) Spectral Density (m$^3$ s$^{-2}$)", fontsize=18)
    plt.title("ROTATIONAL AND DIVERGENT KE SPECTRA: Height Level: %3.1f km" % (zlevels[k]/1000.), fontsize=16)
    
    # KE spectra
    
    axes = fig.add_subplot(gs[0, 1])

    legend='HRRR:black\n\nRRFS:red'

    axes.loglog(avg_grid, tke_hrrr, color='black', linewidth=2.)
    axes.loglog(avg_grid, tke_rrfs, color='red',   linewidth=2.)
    
    axes.set_xlim(2.0/avg_grid.shape[0], 1.0)

    axes.annotate("%s" % legend, xy=(0.10, 0.15), xycoords='axes fraction', color='k',fontsize=18)
    axes.xaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))

    if k < 8:
        ylim = [1.0e10,1.0e18]
    else:
        ylim = [1.0e12,1.0e20]
        
    axes.set_ylim(ylim[0], ylim[1])

    ylabel = 5. * ylim[0]

    xoffset = [0.01, 0.005, 0.0035, 0.001]

    for n, w in enumerate([4.0, 8.0, 12.0, 16.0]):
        axes.axvline(x = (2.0/w), color = 'grey', label = 'axvline - full height')  
        axes.annotate(r"%d$\Delta$x" % w, xy=(2.0/w + xoffset[n], ylabel), xycoords='data', color='k',fontsize=12)

    axes.set_xlabel(r"Wavelength in ($\Delta$x)", fontsize=16)
    axes.set_ylabel(r"E(k) Spectral Density (m$^3$ s$^{-2}$)", fontsize=18)

    plt.title("Total KE SPECTRA ALL: Height Level: %3.1f km" % (zlevels[k]/1000.), fontsize=16)

    plt.savefig("%s/KE_SPECTRA_ALL_%3.1fkm.png" % (out_dir, zlevels[k]/1000.),bbox_inches='tight',dpi=300)
