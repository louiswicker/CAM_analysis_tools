import numpy as np
import xarray as xr
import glob as glob
import os as os
import glob
import sys as sys

import logging

from numpy.fft import fft2, ifft2, fftfreq

from scipy.interpolate import RegularGridInterpolator

from tools.cbook import Dict2Object, open_mfdataset_list, interp_z, write_Z_profile, compute_dbz, compute_thetae

from timeit import default_timer as timer

import warnings
warnings.filterwarnings("ignore")

# def fxn():
#     warnings.warn("deprecated", DeprecationWarning)

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     fxn()
    

_default_file = "cm1out.nc"

_epsil       = 0.608
_Rgas        = 287.04
_rdocp       = 0.286
_cvor        = 2.751
_gravity     = 9.806
_grav        = 9.806
_p00         = 1.0e5


_Cp    = 1004.0
_Cv    = _Cp - _Rgas
_Cvv   = 1424.0
_Lv    = 2.4665e6
_Cpv   = 1885.0

_p00_rdocp   = _p00 ** _rdocp

#--------------------------------------------------------------------------------------------------
#
default_var_map = [        
                   'w',     
                   'accum_prec',
                   ]

#--------------------------------------------------------------------------------------------------
#
class DictAsObject:
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, value)

#--------------------------------------------------------------------------------------------------
#
# READ CM1 FIELDS
#

def read_cm1_fields(path, vars = [''], file_pattern=None, ret_ds=False, 
                    ret_obj=False, ret_beta=False, zinterp=None, unit_test=False):
 
    start0 = timer()
    
    print(f'-'*120)
    
    if file_pattern == None:
        
        # see if the path has the filename on the end....
        
        if os.path.basename(path)[:-3] != ".nc":
            fpath = os.path.join(path, _default_file)

    else:
       fpath = os.path.join(path, file_pattern)
            
    print(f"\n Now reading... {fpath}")

    start = timer()   # time the IO part
    try:
        ds = xr.load_dataset(fpath, decode_times=False)
        ds.load()
    except:
        print(f"Cannot find file in {fpath} exiting")
        sys.exit(-1)
    
    print(f"\n Time for xarray to load:  {fpath}: {timer() - start:.2f} sec ")

    # Variable list processing
    
    if vars != ['']:

        if vars[0] == '+':
            newlist = []
            for var in vars[1:]: 
                newlist.append(var)

            variables = list(set(default_var_map + newlist))

        else:
            variables = vars

    else:
        variables = default_var_map

    # storage bin

    dsout = {}

    start = timer()   # time the IO part
    
    # figure out if we need dbz to be computed
    
    if 'dbz' in variables:
    
        dbz_filename = os.path.join(path, 'dbz.npz')
    
        if os.path.exists(dbz_filename):
            print(f"\n Reading external DBZ file: {dbz_filename}")
            with open(dbz_filename, 'rb') as f:
                dsout['dbz'] = np.load(f)
                
            dsout['cref'] = dsout['dbz'].max(axis=1)

            ret_dbz = False
                        
            variables = list(set(variables + ['dbz']) )
            
        else:
            variables = list(set(variables + ['temp', 'pres', 'qv', 'qc', 'qr']) )
            ret_dbz = True
    else:
        ret_dbz = False

# Add two time variables

    dsout['sec'] = ds.time.values[:]
    dsout['min'] = ds.time.values[:]/60.

# Add some spatial info

    dsout['xc']  = ds.xh.values * 1000.
    dsout['yc']  = ds.yh.values * 1000.
    dsout['zc']  = np.broadcast_to(1000.*ds.zh.values[np.newaxis, :, np.newaxis, np.newaxis], ds.prs.shape)
    dsout['ze']  = np.broadcast_to(1000.*ds.zf.values[np.newaxis, :, np.newaxis, np.newaxis], ds.w.shape)
    dsout['hgt'] = np.broadcast_to(1000.*ds.zh.values[np.newaxis, :, np.newaxis, np.newaxis], ds.prs.shape)

# Add some base state info - these are full 3D/4D arrays

    dsout['base_pii'] = ds.pi0.values
    dsout['base_th']  = ds.th0.values
    dsout['base_qv']  = ds.qv0.values
    dsout['base_prs'] = ds.prs0.values
    dsout['base_t']   = ds.pi0.values * ds.th0.values
    dsout['base_den'] = dsout['base_prs'] / (_Rgas * dsout['base_t'] * (1.0 + _epsil*dsout['base_qv']))

# Some variables we need a lot...

    dsout['pii'] = (ds.prs.values / _p00 )**_rdocp

    print(f"\n Completed setup for:  {fpath} --> time to initialize variables: {timer() - start:.2f} sec ")

    start = timer()   

# Now add variables to dsout created in variable list

    for key in variables:

        if key == 'theta': 
            dsout['theta'] = ds.th.values
            
        if key == 'pert_th': 
            try:
                dsout['pert_th'] = ds.thpert.values
            except:
                dsout['pert_th'] = ds.th.values - dsout['base_th']
            
        if key == 'buoy': 
            dsout['buoy'] = _grav*(ds.thpert.values/dsout['base_th']  \
                          +  _epsil*(ds.qv.values - dsout['base_qv']) \
                          - ds.qc.values - ds.qr.values)

        if key == 'pert_t':
            dsout['pert_t'] = ds.th.values * dsout['base_pii'] - dsout['base_t']
 
        if key == 'u': 
            dsout['u'] = 0.5*(ds.u[:,:,:,1:] + ds.u[:,:,:,:-1]).values

        if key == 'v': 
            dsout['v'] = 0.5*(ds.v[:,:,1:,:] + ds.v[:,:,:-1,:]).values

        if key == 'w': 
            try:
                dsout['w'] = ds.winterp.values
            except:
                dsout['w'] = 0.5*(ds.w[:,1:,:,:] + ds.w[:,:-1,:,:]).values

            dsout['wmax'] = dsout['w'].max(axis=1)

        if key == 'vvort': 
            dsout['vvort'] = np.zeros_like(ds.thpert.values)

        if key == 'pres':
            dsout['pres'] = ds.prs.values

        if key == 'pert_p':
            dsout['pert_p'] = ds.prspert.values

        if key == 'base_p':
            dsout['base_p'] = ds.prs[0,:,-1,-1].values

        if key == 'qv':
            dsout['qv'] = ds.qv.values

        if key == 'qc':
            dsout['qc'] = ds.qc.values

        if key == 'qr':
            dsout['qr'] = ds.qr.values

        if key == 'den':
            dsout['den'] = ds.prs.values / (_Rgas*(ds.th.values*dsout['pii']))

        if key == 'rwqv':
            den           = ds.prs.values / (_Rgas*(ds.th.values*dsout['pii']))
            w             = ds.winterp.values
            dsout['rwqv'] = den * w *  ds.qv.values
       
        if key == 'rw':
            dsout['rw']   = den * w
                            
        if key == 'rho':
            dsout['rho'] = ds.prs.values / (_Rgas*(ds.th.values*dsout['pii']*(1.0+_epsil*ds.qv.values)*(1.0 - ds.qc.values - ds.qr.values)))
            
        if key == 'temp':
            dsout['temp'] = ds.th.values*dsout['pii']

        if key == 'dwdt':
            tmp             = ds.wb_pgrad.values + ds.wb_buoy.values
            dsout['dwdt']   = 0.5*(tmp[:,1:,:,:] + tmp[:,:-1,:,:])
            try:
                dsout['pgradb'] = ds.pgradb.values
            except:
                print(" -->Could not find buoyant pressure gradient decomp\n")

        if key == 'accum_prec':
            dsout['accum_prec'] = 10.*ds.rain.values  # convert CM to MM

        if key == 'div2d':
            print(" -->Computing finite difference 2D divergence\n")
            u = ds.u.values
            v = ds.v.values
            dx = ds.xh[1].values - ds.xh[0].values
            dsout['div2d'] = (u[:,:,:,1:] - u[:,:,:,:-1] + v[:,:,1:,:] - v[:,:,:-1,:]) / dx

        if key == 'uns_div2d':
            print(" -->Computing unstaggered finite difference 2D divergence\n")
            us = ds.u.values
            u  = 0.5*(us[:,:,:,1:] + us[:,:,:,:-1])

            vs = ds.v.values
            v  = 0.5*(vs[:,:,1:,:] + vs[:,:,:-1,:])

            dx = ds.xh[1].values - ds.xh[0].values

            dudx = np.gradient(u, axis=3)
            dvdy = np.gradient(v, axis=2)

            dsout['uns_div2d'] = (dudx + dvdy) / dx

        if key == 'fft_div2d':

            print(" -->Computing FFT 2D divergence\n")

            x1 = ds.xh[:-1].values
            y1 = ds.yh[:-1].values
            Nx = x1.shape[0]
            Ny = y1.shape[0]

            if Nx != Ny:
                print("\n Warning, Nx != Ny, not sure if fft works!....\n")

            kx = fftfreq(Nx) * 2*np.pi * Nx / (x1[-1] - x1[0])
            ky = fftfreq(Ny) * 2*np.pi * Ny / (y1[-1] - y1[0])
            Kx, Ky  = np.meshgrid(kx, ky, indexing='ij')
            wavenumbers = np.stack((Kx, Ky))
            derivative_op = 1j * wavenumbers

            us = ds.u.values
            u  = 0.5*(us[:,:,:,1:] + us[:,:,:,:-1])[:,:,:-1,:-1]

            vs = ds.v.values
            v  = 0.5*(vs[:,:,1:,:] + vs[:,:,:-1,:])[:,:,:-1,:-1]

            div2d = np.zeros_like(u)

            for n in np.arange(u.shape[0]):
                for k in np.arange(ds.nz):
                    div2d[n,k] = ifft2(1j * Kx * fft2(u[n,k])) \
                               + ifft2(1j * Ky * fft2(v[n,k]))

            dsout['fft_2d_div'] = div2d.real

        if key == 'total_e':
            print(" -->Computing Total energy \n")

            rv          = ds.qv.values
            rl          = ds.qc.values + ds.qr.values
            temperature = ds.th.values * (ds.prs.values / _p00 )**_rdocp
            dens        = ds.prs.values / (_Rgas*ds.th.values*dsout['pii'])

            dsout['total_e'] = dens * ( _Cv*temperature + _Cvv*rv*temperature + _Cpv*rl*temperature - _Lv*rl + _grav*(1.0+rv+rl)*dsout['zc'] )

        if key == 'thetae':

            print(" -->Computing ThetaE \n")
            
            dsout['qv']     = ds.qv.values 
            dsout['temp']   = ds.th.values*dsout['pii']
            dsout['pres']   = ds.prs.values
            dsout['thetae'] = compute_thetae(dsout)

    if unit_test:
        write_Z_profile(dsout, model='CM1', vars=variables, loc=(10,-1,1))

    if ret_dbz:
        dsout = compute_dbz(dsout, version=2)
        with open(dbz_filename, 'wb') as f:  np.save(f, dsout['dbz'])
        dsout['cref'] = dsout['dbz'].max(axis=1)

        variables = list(set(variables + ['dbz']) )
        
    if ret_beta:      # returning the mass*acceleration from Beta, so divide by density

        # read in BETA
        
        print(f"\n Reading BETA from {os.path.join(path, 'w_b_accel.nc')}")

        dsbeta = xr.load_dataset(os.path.join(path, "w_b_accel.nc"), decode_times=False)

        # read forcing in as well.

        print(f"\n Reading density from {os.path.join(path, 'total_den.nc')}")

        dsrho = xr.load_dataset(os.path.join(path, "total_den.nc"), decode_times=False)

        den1d = dsrho.den.values[0,:,0,0]
        den3d = np.broadcast_to(den1d[np.newaxis, :, np.newaxis, np.newaxis], dsrho.den.shape)
 
        dsout['beta']  = (dsbeta.beta.values / den3d)
        dsout['rho_p'] = (dsrho.den.values - den3d)

        dsrho.close()
        dsbeta.close()

        variables = list(set(variables + ['beta','rho_p']) )
    
    print(f"\n Completed reading in:  {fpath} --> time to process variables: {timer()  - start:.2f} sec ")

    if zinterp is None:
        pass
        
    else:
        print(f"\n Interpolating fields to single column z-grid: {fpath} \n")

        start = timer()

        

        for key in variables:
             if dsout[key].ndim == dsout['zc'].ndim:
                try:
                    tmp =  interp_z(dsout[key], dsout['zc'], zinterp)
                    dsout[key] = tmp
                except ValueError:
                    print(f" Interpolation could not be done on {key}, as shape is {dsout[key].shape}")
            
        new_shape   = [dsout['zc'].shape[0],zinterp.shape[0],dsout['zc'].shape[2],dsout['zc'].shape[3]]
        dsout['zc'] = np.broadcast_to(zinterp[np.newaxis, :, np.newaxis, np.newaxis], new_shape)
        
        print(f" Interpolated completed, Total CPU:  {path}: {timer() - start:.2f} sec \n") 
                
    # Finish

    ds.close()

    print(f" Total time for processing file:  {path}: --> {timer() - start0:.2f} sec \n") 
    print(f'-'*120)

    if ret_obj:
        dsout = Dict2Object(dsout)

    if ret_ds:
        return dsout, ds
    else:
        return dsout

#########################################################################################
#
# Routine for fast variable read 
#
#

def read_cm1_w(path, var = 'w', file_pattern=None, netCDF=False):

    from netCDF4 import MFDataset, Dataset
 
    print(f'-'*120)
    
    if file_pattern == None:
        
        # see if the path has the filename on the end....
        
        if os.path.basename(path)[:-3] != ".nc":
            fpath = os.path.join(path, _default_file)
            
    else:
       fpath = os.path.join(path, file_pattern)
            
    print(f" Now Reading:  {fpath}")
       
    try:
        fobj = Dataset(fpath)
    except:
        print(f"Cannot find the file in {fpath}, exiting")
        sys.exit(-1)

# storage bin

    dsout = {}

# Add two time variables

    dsout['sec'] = fobj.variables['time'][:]
    dsout['min'] = fobj.variables['time'][:]/60.

# Add some spatial info

    zc           = fobj.variables['zh'][:]
    theta        = fobj.variables['th'][...]

    dsout['xc']  = 1000*fobj.variables['xh'][:]
    dsout['yc']  = 1000*fobj.variables['yh'][:]
    dsout['zc']  = 1000.*np.broadcast_to(zc[np.newaxis, :, np.newaxis, np.newaxis], theta.shape)
    dsout['hgt'] = 1000.*np.broadcast_to(zc[np.newaxis, :, np.newaxis, np.newaxis], theta.shape)

    if var == 'w': 

        try:
            dsout['w'] = fobj.variables['winterp'][...]
        except:
            dsout['w'] = 0.5*(fobj.variables['w'][:,1:,:,:] + fobj.variables['w'][:,:-1,:,:]).values

        print(f" Completed reading: {fpath} \n")

    return DictAsObject(dsout)
