import numpy as np
import netCDF4 as ncdf
import matplotlib.pyplot as plt
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

_Rgas       = 287.04
_gravity    = 9.806
_grav       = 9.806
_default_file = "atmos_hifreq.nc"

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
                   'pres',   
                   'pert_p',   
                   'pii',     
                   'accum_prec',
                   'theta',    
                   'pert_th',   
                   ]

#--------------------------------------------------------------------------------------------------
#
# READ SOLO FIELDS
#

def read_solo_fields(path, vars = [''], file_pattern=None, ret_dbz=False, ret_ds=False, unit_test=False):
        
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
            print("Cannot find file in %s, exiting"  % path)
            sys.exit(-1)

    else:
        ds = xr.load_dataset(os.path.join(path, file_pattern), decode_times=False)
        
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
            dsout['theta'] = ds.theta.values[:,::-1,:,:]
            
        if key == 'temp': 
            pii           = (ds.nhpres.values[:,::-1,:,:] / 100000.)**0.286
            dsout['temp'] = pii*ds.theta.values[:,::-1,:,:]
            
        if key == 'pert_t': 
            pii    = (ds.nhpres.values[:,::-1,:,:] / 100000.)**0.286
            base_t = pii[0,:,-1,-1] * ds.theta.values[0,::-1,-1,-1]
            dsout['pert_t'] = pii*ds.theta.values[:,::-1,:,:] - np.broadcast_to(base_t[np.newaxis, :, np.newaxis, np.newaxis], ds.theta.shape) 
            
        if key == 'pert_th': 
            pii      = (ds.nhpres.values[:,::-1,:,:] / 100000.)**0.286
            theta    = ds.theta.values[:,::-1,:,:]
            base_th  = theta[0,:,-1,-1]/pii[0,:,-1,-1]
            dsout['pert_th'] = theta - np.broadcast_to(base_th[np.newaxis, :, np.newaxis, np.newaxis], theta.shape) 

        if key == 'buoy':
            pii      = (ds.nhpres.values[:,::-1,:,:] / 100000.)**0.286
            theta    = ds.theta.values[:,::-1,:,:]
            base_th  = theta[0,:,-1,-1]
            base_th  = np.broadcast_to(base_th[np.newaxis, :, np.newaxis, np.newaxis], theta.shape) 
            pert_th  = theta - base_th
            qv       = ds.spfh.values[:,::-1,:,:] / (1.0 + ds.spfh.values[:,::-1,:,:])  # convert to mix-ratio
            base_qv  = qv[0,:,-1,-1]
            base_qv  = np.broadcast_to(base_qv[np.newaxis, :, np.newaxis, np.newaxis], theta.shape) 
            pert_qv  = qv - base_qv
            dsout['buoy'] = 9.806*(pert_th/base_th + 0.61*pert_qv \
                          - ds.clwmr.values[:,::-1,:,:] - ds.rwmr.values[:,::-1,:,:])
            
        if key == 'u': 
            dsout['u'] = ds.ugrd.values[:,::-1,:,:]

        if key == 'v': 
            dsout['v'] = ds.vgrd.values[:,::-1,:,:]

        if key == 'w': 
            dsout['w'] = ds.w.values[:,::-1,:,:]

        if key == 'dzdt': 
            tmp           = 0.5*(ds.dzdt.values[:,1:,:,:] + ds.dzdt.values[:,:-1,:,:])
            dsout['dzdt'] = tmp[:,::-1,:,:]

        if key == 'vvort': 
            dsout['vvort'] = ds.rel_vort.values[:,::-1,:,:]

        if key == 'hgt': 
            ze = np.cumsum(ds.delz.values[:,::-1,:,:], axis=1)
            dsout['hgt'] = np.zeros_like(ze)
            dsout['hgt'][:,0,:,:]  = 0.5*ze[:,0,:,:]
            dsout['hgt'][:,1:,:,:] = 0.5*(ze[:,:-1,:,:] + ze[:,1:,:,:])

        if key == 'pres':
            dsout['pres'] = ds.nhpres.values[:,::-1,:,:]

        if key == 'pert_p':    # code from L Harris Jupyter notebook
          # ptop       = ds.phalf[0]
          # phalf      = ds.phalf[1:]
          # pfull      = (ds.delp.sum(dim='pfull') + ptop).values[:,::-1,:,:]
          # pfull_ref  = np.broadcast_to(pfull[0,:,0,0][np.newaxis, :, np.newaxis, np.newaxis], ds.nhpres.shape)
          # p_from_qv  = ((ds.spfh)*ds.delp).sum(dim='pfull').values[:,::-1,:,:]
          # p_from_qp  = ds.qp.sum(dim='pfull').values[:,::-1,:,:]
            
          # dsout['pert_p']  = pfull - pfull_ref - p_from_qv - p_from_qp            
            dsout['pert_nh'] = ds.nhpres[:,::-1,:,:].values
            dsout['pert_p']  = ds.nhpres[:,::-1,:,:].values

        if key == 'base_p':
            dsout['base_p'] = np.broadcast_to(ds.pfull.values[::-1][np.newaxis, :, np.newaxis, np.newaxis], ds.nhpres.shape)

        if key == 'dpdz':
            dsout['dpdz'] = ds.vaccel.values[:,::-1,:,:]

        if key == 'qv':
            dsout['qv'] = ds.spfh.values[:,::-1,:,:] / (1.0 + ds.spfh.values[:,::-1,:,:])  # convert to mix-ratio

        if key == 'qc':
            dsout['qc'] = ds.clwmr.values[:,::-1,:,:]

        if key == 'qr':
            dsout['qr'] = ds.rwmr.values[:,::-1,:,:]

        if key == 'den':
            pii          = (ds.nhpres.values[:,::-1,:,:] / 100000.)**0.286
            dsout['den'] = ds.press.values[:,::-1,:,:] / (_Rgas * pii*ds.theta.values[:,::-1,:,:])

        if key == 'pii':
            dsout['pii'] = (ds.nhpres.values[:,::-1,:,:] / 100000.)**0.286

        if key == 'accum_prec':
            try:
                dsout['accum_prec'] = ds.rain_k.values[:,:,:]  # original name
            except:
                dsout['accum_prec'] = ds.accumulated_rain.values[:,:,:] # new name

    if unit_test:
        write_Z_profile(dsout, model='SOLO', vars=variables, loc=(10,-1,1))
        
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