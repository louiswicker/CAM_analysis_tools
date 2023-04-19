import numpy as np
import xarray as xr
import glob as glob
import os as os
import glob
import sys as sys
from tools.cbook import write_Z_profile, compute_dbz

import warnings
warnings.filterwarnings("ignore")

# def fxn():
#     warnings.warn("deprecated", DeprecationWarning)

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     fxn()
    
_nthreads = 2

_Rgas         = 287.04
_gravity      = 9.806
_grav         = 9.806
_default_file = "cm1out.nc"

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
                   
#--------------------------------------------------------------------------------------------------
def open_mfdataset_list(data_dir, pattern):

    """
    Use xarray.open_mfdataset to read multiple netcdf files from a list.
    """

    filelist = os.path.join(data_dir,pattern)
    return xr.open_mfdataset(filelist, parallel=True)
#--------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------
#
# READ CM1 FIELDS
#

def read_cm1_fields(path, vars = [''], file_pattern=None, ret_ds=False, 
                    ret_dbz=False, unit_test=False):
 
    if file_pattern == None:
        # see if the path has the filename on the end....
        if os.path.basename(path)[:-3] != ".nc":
            path = os.path.join(path, _default_file)
            print(f'-'*120,'\n')
            print(" Added default filename to path input:  %s" % path)

        print(f'-'*120,'\n')
        print(" Reading:  %s \n" % path)
        try:
                ds = xr.load_dataset(path, decode_times=False)
        except:
                print("Cannot find the file in %s, exiting"  % path)
                sys.exit(-1)
    else:
        ds = open_mfdataset_list(run_dir,  file_pattern)
        
    if vars != ['']:
        variables = vars
    else:
        variables = default_var_map

    # storage bin

    dsout = {}
        
    # figure out if we need dbz to be computed
    
    if ret_dbz:
    
        dbz_filename = os.path.join(os.path.dirname(path), 'dbz.npz')
    
        if os.path.exists(dbz_filename):
            print("\n Reading external DBZ file: %s\n" % dbz_filename)
            with open(os.path.join(os.path.dirname(path), 'dbz.npz'), 'rb') as f:
                dsout['dbz'] = np.load(f)
                
            ret_dbz = False
                        
            # for n in np.arange(dsout['dbz'].shape[0]):
            #     print(n, dsout['dbz'][n].max())
            
        else:
            variables = list(set(variables + ['temp','pres', 'qv', 'qc', 'qr']) )

    for key in variables:

        if key == 'theta': 
            dsout['theta'] = ds.th.values
            
        if key == 'pert_th': 
            dsout['pert_th'] = ds.thpert.values
            
        if key == 'pert_t':
            base_pii = ds.pi0.values  
            base_th0 = ds.th0.values
            base_t0  = base_th0*base_pii
            pii      = (ds.prs.values / 100000.)**0.286
            temp     = ds.th.values * pii
            dsout['pert_t'] = temp - base_t0
 
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
            dsout['den'] = ds.prs.values / (_Rgas*ds.th.values*pii)
            
        if key == 'temp':
            pii  = (ds.prs.values / 100000.)**0.286
            dsout['temp'] = ds.th.values*pii

        if key == 'pii':
            dsout['pii'] = (ds.prs.values / 100000.)**0.286

        if key == 'accum_prec':
            dsout['accum_prec'] = 10.*ds.rain.values  # convert CM to MM
            
    if unit_test:
        write_Z_profile(dsout, model='CM1', vars=variables, loc=(10,-1,1))

    if ret_dbz:
        dsout = compute_dbz(dsout, version=2)
        with open(dbz_filename, 'wb') as f:  np.save(f, dsout['dbz'])
        
    print(" Completed reading in:  %s \n" % path)
    print(f'-'*120)

    if ret_ds:
        ds.close()
        return dsout, ds
    else:
        ds.close()
        return dsout
 
