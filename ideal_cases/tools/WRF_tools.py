import numpy as np
import netCDF4 as ncdf
import xarray as xr
import glob as glob
import os as os
import glob
import sys as sys

from tools.cbook import Dict2Object, open_mfdataset_list, interp_z, write_Z_profile, compute_dbz, compute_thetae

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
_default_file = "wrfout_d01_0001-01-01_00:00:00"

default_var_map = [        
                   'w',     
                   'pert_p',   
                   'accum_prec',
                   'pert_th',   
                   ]
                   
#--------------------------------------------------------------------------------------------------
#
# WRF file version of mfdataset with xarray
#

def open_mfdataset_list(data_dir, pattern, skip=0):
        """
        Use xarray.open_mfdataset to read multiple netcdf files from a list.
        """
        filelist = os.path.join(data_dir,pattern)
        
        if skip > 0:
                filelist = filelist[0:-1:skip]
                
        return xr.load_mfdataset(filelist, combine='nested', concat_dim=['Time'], parallel=True)

#
#--------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------
#
# READ WRF FIELDS
#

def read_wrf_fields(path, vars = [''], file_pattern=None, ret_ds=False, 
                    ret_dbz=False, unit_test=False):
 
    if file_pattern == None:
    
    # see if the path has the filename on the end....
        if os.path.basename(path)[0:6] != "wrfout":
            path = os.path.join(path, _default_file)
            print(f'-'*120,'\n')
            print(" Added default filename to path input:  %s" % path)

        print(f'-'*120,'\n')
        print(" Reading:  %s \n" % path)
        try:
            ds = xr.load_dataset(path, decode_times=False)
        except:
            print("Cannot find file in %s, exiting"  % path)
            sys.exit(-1)

    else:
        ds   = open_mfdataset_list(path, file_pattern)
        
    # Process variable input

    if vars != ['']:
        variables = vars
    else:
        variables = default_var_map

    # set up output storage dict

    dsout = {}
        
    # figure out if we need dbz to be computed
    
    if ret_dbz:
    
        dbz_filename = os.path.join(os.path.dirname(path), 'dbz.npz')
    
        if os.path.exists(dbz_filename):
            print("\n Reading external DBZ file: %s\n" % dbz_filename)
            with open(os.path.join(os.path.dirname(path), 'dbz.npz'), 'rb') as f:
                dsout['dbz'] = np.load(f)
                
            ret_dbz = False

            dsout['cref'] = dsout['dbz'].max(axis=1)
                        
            # for n in np.arange(dsout['dbz'].shape[0]):
            #     print(n, dsout['dbz'][n].max())
            
        else:
            variables = list(set(variables + ['temp', 'pres', 'qv', 'qc', 'qr']) )

    for key in variables:

        if key == 'theta': 
            tmp1  = ds.T.values + 300.
            dsout['theta'] = tmp1
            
        if key == 'pert_th': 
            base_th = (ds.T[0,:,-1,-1].values+300.)
            dsout['pert_th'] = (ds.T.values+300.) - np.broadcast_to(base_th[np.newaxis, :, np.newaxis, np.newaxis], ds.T.shape)
            
        if key == 'buoy':
            base_th = (ds.T[0,:,-1,-1].values+300.)
            base_th = np.broadcast_to(base_th[np.newaxis, :, np.newaxis, np.newaxis], ds.T.shape)
            thpert  = (ds.T.values+300.) - base_th
            base_qv = (ds.QVAPOR[0,:,-1,-1].values)
            qvpert  = (ds.QVAPOR.values) - np.broadcast_to(base_qv[np.newaxis, :, np.newaxis, np.newaxis], ds.T.shape)

            dsout['buoy'] = 9.806*(thpert/base_th + 0.61*qvpert - ds.QCLOUD.values - ds.QRAIN.values)

        if key == 'u': 
            u      = ds.U.values
            dsout['u'] = 0.5*(u[:,:,:,1:] + u[:,:,:,:-1])

        if key == 'v': 
            v      = ds.V.values
            dsout['v'] = 0.5*(v[:,:,1:,:] + v[:,:,:-1,:])

        if key == 'w': 
            dsout['w'] = 0.5*(ds.W.values[:,1:,:,:] + ds.W.values[:,:-1,:,:])
            dsout['wmax'] = dsout['w'].max(axis=1)

        if key == 'vvort': 
            dsout['vvort'] = np.zeros_like(ds.T.values)

        if key == 'hgt': 
            z = (ds.PH.values + ds.PHB.values) / _grav
            dsout['hgt'] = 0.5*(z[:,1:,:,:] + z[:,:-1,:,:])

        if key == 'pres':
            dsout['pres'] = ds.P.values + ds.PB.values

        if key == 'pert_p':
        #   pertp = ds.P.values
        #   meanp = pertp.mean(axis=(1,2,3))
        #   dsout['pert_p'] = ds.P.values - np.broadcast_to(meanp[:, np.newaxis, np.newaxis, np.newaxis], ds.T.shape)
            dsout['pert_p'] = ds.P.values

        if key == 'dpdz':
            dsout['dpdz'] = ds.PB.values

        if key == 'qv':
            dsout['qv'] = ds.QVAPOR.values

        if key == 'qc':
            dsout['qc'] = ds.QCLOUD.values

        if key == 'qr':
            dsout['qr'] = ds.QRAIN.values

        if key == 'den':
            pii  = ((ds.P.values + ds.PB.values) / 100000.)**0.286
            dsout['den'] = (ds.P.values + ds.PB.values) / (_Rgas * (ds.T.values+300.)*pii)

        if key == 'temp':
            pii  = ((ds.P.values + ds.PB.values) / 100000.)**0.286
            dsout['temp'] = (ds.T.values+300.)*pii
            
        if key == 'pert_t':
            pii       = ((ds.P.values + ds.PB.values) / 100000.)**0.286
            base_pii  = (ds.PB[0,:,-1,-1].values / 100000.)**0.286
            base_temp = (ds.T[0,:,-1,-1].values+300.)*base_pii
            dsout['pert_t'] = (ds.T.values+300.)*pii - np.broadcast_to(base_temp[np.newaxis, :, np.newaxis, np.newaxis], ds.T.shape)

        if key == 'pii':
            dsout['pii'] = ((ds.P.values + ds.PB.values) / 100000.)**0.286

        if key == 'accum_prec':
            dsout['accum_prec'] = ds.RAINNC.values  

    if unit_test:
        write_Z_profile(dsout, model='WRF', vars=variables, loc=(10,-1,1))
        
    if ret_dbz:
        dsout = compute_dbz(dsout, version=2)
        with open(dbz_filename, 'wb') as f:  np.save(f, dsout['dbz'])
        dsout['cref'] = dsout['dbz'].max(axis=1)
        
    print(" Completed reading in:  %s \n" % path)
    print(f'-'*120)

    if ret_ds:
        ds.close()
        return dsout, ds
    else:
        ds.close()
        return dsout
