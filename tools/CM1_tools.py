import numpy as np
import xarray as xr
import glob as glob
import os as os
import glob
import sys as sys
from cbook import write_Z_profile


import warnings
warnings.filterwarnings("ignore")

# def fxn():
#     warnings.warn("deprecated", DeprecationWarning)

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     fxn()
    
_nthreads = 2

_Rgas       = 287.04
_gravity    = 9.806
_grav       = 9.806

default_var_map = [        
                   'temp',   
                   'hgt',  
                   'w',     
                   'u',    
                   'v',   
                   'vvort', 
                   'qv',     
                   'qc',    
                   'qr',     
                   'press',   
                   'pert_p',   
                   'pii',     
                   'accum_prec',
                   'theta',    
                   'pert_th',   
                   ]
#==========================================================================================================
#
# READ CM1 FIELDS
#

def read_cm1_fields(path, vars = [''], file_pattern=None, unit_test=False):
 
    #--------------------------------------------------------------------------------------------------
    def open_mfdataset_list(data_dir, pattern):
        """
        Use xarray.open_mfdataset to read multiple netcdf files from a list.
        """
        filelist = os.path.join(data_dir,pattern)
        return xr.open_mfdataset(filelist, parallel=True)
    #--------------------------------------------------------------------------------------------------
    
    if file_pattern == None:
        print(f'-'*120,'\n')
        print(" Reading:  %s \n" % path)
        print(f'-'*120,'\n')
        ds = xr.open_dataset(path, decode_times=False)
    else:
        ds = open_mfdataset_list(run_dir,  file_pattern)
        
    if vars != ['']:
        variables = vars
    else:
        variables = default_var_map

    dsout = {}

    for key in variables:

        if key == 'theta': 
            dsout['theta'] = ds.th0.values + ds.thpert.values
            
        if key == 'pert_th': 
            dsout['pert_th'] = ds.thpert.values
            
        if key == 'u': 
            u      = ds.u.values
            dsout['u'] = 0.5*(u[:,:,:,1:] + u[:,:,:,:-1])

        if key == 'v': 
            v      = ds.v.values
            dsout['v'] = 0.5*(v[:,:,1:,:] + v[:,:,:-1,:])

        if key == 'w': 
            dsout['w'] = ds.winterp.values

        if key == 'vvort': 
            dsout['vvort'] = np.zeros_like(ds.thpert.values)

        if key == 'hgt': 
            dsout['hgt']   = np.broadcast_to(1000.*ds.zh.values[np.newaxis, :, np.newaxis, np.newaxis], ds.winterp.shape)

        if key == 'pres':
            dsout['pres'] = ds.prs.values

        if key == 'pert_p':
            dsout['pert_p'] = ds.prspert.values

        if key == 'base_p':
            dsout['base_p'] = ds.prs0.values

        if key == 'qv':
            dsout['qv'] = ds.qv.values

        if key == 'qc':
            dsout['qc'] = ds.qc.values

        if key == 'qr':
            dsout['qr'] = ds.qr.values

        if key == 'den':
            pii  = (ds.prs.values / 100000.)**0.286
            dsout['den'] = ds.prs.values / (287.04*(ds.th0.values + ds.thpert.values)*pii)
            
        if key == 'temp':
            pii  = (ds.prs.values / 100000.)**0.286
            dsout['temp'] = (ds.th0.values + ds.thpert.values)*pii


        if key == 'pii':
            dsout['pii'] = (ds.prs.values / 100000.)**0.286

        if key == 'accum_prec':
            dsout['accum_prec'] = 10*ds.rain.values
            
    if unit_test:
        write_Z_profile(dsout, model='CM1', vars=variables, loc=(10,-1,1))

    ds.close()
        
    return dsout
