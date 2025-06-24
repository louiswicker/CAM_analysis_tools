import numpy as np
import netCDF4 as ncdf
import matplotlib.pyplot as plt
import xarray as xr
import glob as glob
import os as os
import glob
import sys as sys

from numpy.fft import fftn, ifftn, fftfreq

from tools.cbook import Dict2Object, open_mfdataset_list, interp_z, write_Z_profile, compute_dbz, compute_thetae

import warnings
warnings.filterwarnings("ignore")

# def fxn():
#     warnings.warn("deprecated", DeprecationWarning)

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     fxn()
    
_nthreads = 2

# from FV3_Solo/SHiELD_SRC/FMS/constants/geos_constants.fh

RGAS   = 8314.47 / 28.965           
KAPPA  = RGAS/(3.5*RGAS)         
CP_AIR = RGAS/KAPPA        

_Rgas  = RGAS
_grav  = 9.806
_Cp    = CP_AIR
_Cv    = CP_AIR - RGAS
_Cvv   = 1424.0
_Lv    = 2.4665e6
_Cpv   = 1885.0
_epsil = 0.608


_default_file = "atmos_hifreq.nc"

#--------------------------------------------------------------------------------------------------
#
default_var_map = [                # do not include coordinates in this list, automatically added.       
                   'w',     
                   'pert_p',   
                   'accum_prec',
                   'pert_th',   
                   ]

#--------------------------------------------------------------------------------------------------
#
class DictAsObject:
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, value)

#--------------------------------------------------------------------------------------------------
#
# READ SOLO FIELDS
#

def read_solo_fields(path, vars = [''], file_pattern=None, ret_dbz=False, 
                     ret_ds=False, ret_obj=False, ret_beta=False, zinterp=None, unit_test=False):
        
    print(f'-'*120)
    
    if file_pattern == None:
        
        # see if the path has the filename on the end....
        
        if os.path.basename(path)[:-3] != ".nc":
            fpath = os.path.join(path, _default_file)

    else:
       fpath = os.path.join(path, file_pattern)
            
    print(f" Now reading... {fpath}")
        
    try:
        ds = xr.load_dataset(fpath, decode_times=False)
    except:
        print(f"Cannot find file in {fpath} exiting"  % path)
        sys.exit(-1)

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
        
    # figure out if we need dbz to be computed
    
#   if ret_dbz:
#   
#       dbz_filename = os.path.join(os.path.dirname(path), 'dbz.npz')
#   
#       if os.path.exists(dbz_filename):
#           print("\n Reading external DBZ file: %s\n" % dbz_filename)
#           with open(os.path.join(os.path.dirname(path), 'dbz.npz'), 'rb') as f:
#               dsout['dbz'] = np.load(f)
#               
#           dsout['cref'] = dsout['dbz'].max(axis=1)
#
#           ret_dbz = False
#                       
#          #for n in np.arange(dsout['dbz'].shape[0]):
#          #    print(n, dsout['dbz'][n].max())
#           
#       else:
#           variables = list(set(variables + ['temp','pres', 'qv', 'qc', 'qr']) )

# Add two time variables

    dsout['sec'] = ds.time.values[:]
    dsout['min'] = ds.time.values[:]/60.

# Always add coordinates to data structure

    dsout['xc'] = ds.grid_xt.values * 3000.
    dsout['yc'] = ds.grid_yt.values * 3000.

    ze = np.cumsum(ds.delz.values[:,::-1,:,:], axis=1)

    dsout['ze']  = ze
    dsout['zc']  = np.zeros_like(ze)
    dsout['zc'][:,0,:,:]  = 0.5*ze[:,0,:,:]
    dsout['zc'][:,1:,:,:] = 0.5*(ze[:,:-1,:,:] + ze[:,1:,:,:])
    dsout['hgt'] = dsout['zc']

# Some variables we need a lot...the pressure calcs are pulled from Harris Jupyter notebook
    
    ptop              = ds.phalf[0]
    phalf             = ds.phalf[1:]
    pfull             = (ds.delp.cumsum(dim='pfull') + ptop).values
    dsout['prs_full'] = pfull[:,::-1,:,:]
    p_from_qv         = ((ds.sphum)*ds.delp).cumsum(dim='pfull').values
    p_from_qp         = ds.rwmr.cumsum(dim='pfull').values + ds.clwmr.cumsum(dim='pfull').values
    dsout['prs_dry']  = (pfull - (p_from_qv - p_from_qp))[:,::-1,:,:]
    dsout['pii']      = (dsout['prs_dry'] / 100000.)**0.286

    for key in variables:

        if key == 'theta': 
            dsout['theta'] = ds.theta.values[:,::-1,:,:]
            
        if key == 'temp': 
            dsout['temp'] = dsout['pii']*ds.theta.values[:,::-1,:,:]
            
        if key == 'pert_t': 
            base_t = pii[0,:,-1,-1] * ds.theta.values[0,::-1,-1,-1]
            dsout['pert_t'] = dsout['pii']*ds.theta.values[:,::-1,:,:] \
                            - np.broadcast_to(base_t[np.newaxis, :, np.newaxis, np.newaxis], ds.theta.shape) 
            
        if key == 'pert_th': 
            theta    = ds.theta.values[:,::-1,:,:]
            base_th  = theta[0,:,-1,-1]
            dsout['pert_th'] = theta - np.broadcast_to(base_th[np.newaxis, :, np.newaxis, np.newaxis], theta.shape) 

        if key == 'buoy':
            theta    = ds.theta.values[:,::-1,:,:]
            base_th  = theta[0,:,-1,-1]
            base_th  = np.broadcast_to(base_th[np.newaxis, :, np.newaxis, np.newaxis], theta.shape) 
            pert_th  = theta - base_th
            qv       = ds.sphum.values[:,::-1,:,:] / (1.0 + ds.sphum.values[:,::-1,:,:])  # convert to mix-ratio
            base_qv  = qv[0,:,-1,-1]
            base_qv  = np.broadcast_to(base_qv[np.newaxis, :, np.newaxis, np.newaxis], theta.shape) 
            pert_qv  = qv - base_qv
            dsout['buoy'] = _grav*(pert_th/base_th + 0.61*pert_qv - ds.clwmr.values[:,::-1,:,:] - ds.rwmr.values[:,::-1,:,:])
            
        if key == 'dpstar':
            dsout['dpstar'] = ds.delp.values[:,::-1,:,:]

        if key == 'ppedge':
            try:
                tmp             = ds.wforcn[:,::-1,:,:].values
                dsout['ppedge'] = np.zeros((tmp.shape[0], tmp.shape[1]+1, tmp.shape[2], tmp.shape[3]))
                dsout['ppedge'][:,:-1,:,:] = tmp
            except:
                dsout['ppedge'] = np.zeros_like(ds.nhpres)

        if key == 'dwdt':
            try:
                dsout['dwdt']   = ds.wforcn[:,::-1,:,:]
            except:
                dpstar           = ds.delp.values[:,::-1,:,:]
                pnh              = ds.pnhpres.values[:,::-1,:,:]
                dpnh             = np.zeros_like(pnh)
                dpnh[:,1:-1,:,:] = 0.5*(pnh[:,2:,:,:] - pnh[:,:-2,:,:])
                dsout['dwdt']    = -_grav*(dpnh / dpstar)

        if key == 'u': 
            dsout['u'] = ds.ugrd.values[:,::-1,:,:]

        if key == 'v': 
            dsout['v'] = ds.vgrd.values[:,::-1,:,:]

        if key == 'w': 
            dsout['w'] = ds.w.values[:,::-1,:,:]

        if key == 'dzdt': 
            dsout['dzdt'] = 0.5*(ds.dzdt.values[:,1:,:,:] + ds.dzdt.values[:,:-1,:,:])[:,::-1,:,:]

        if key == 'vvort': 
            dsout['vvort'] = ds.rel_vort.values[:,::-1,:,:]

        if key == 'pres':
            dsout['pres'] = ds.nhpres.values[:,::-1,:,:]

        if key == 'pert_p':    # code from L Harris Jupyter notebook
            ptop       = ds.phalf[0]
            pfull      = (ds.delp.cumsum(dim='pfull') + ptop).values
            pfull_ref  = np.broadcast_to(pfull[0,:,0,0][np.newaxis, :, np.newaxis, np.newaxis], ds.nhpres.shape)
            p_from_qv  = ((ds.sphum)*ds.delp).cumsum(dim='pfull').values
            p_from_qp  = ds.rwmr.cumsum(dim='pfull').values + ds.clwmr.cumsum(dim='pfull').values
            
            dsout['pert_lucas'] = (pfull - pfull_ref - (p_from_qv - p_from_qp))[:,::-1,:,:]
            dsout['pert_p'] = ds.nhpres_pert[:,::-1,:,:].values 
#           dsout['pert_p']  = ds.nhpres[:,::-1,:,:].values - pfull_ref

        if key == 'base_p':
            dsout['base_p'] = np.broadcast_to(dsout['prs_dry'][np.newaxis, :, np.newaxis, np.newaxis], ds.nhpres.shape)

        if key == 'dpdz':
            dsout['dpdz'] = ds.vaccel.values[:,::-1,:,:]

        if key == 'qv':
            dsout['qv'] = ds.sphum.values[:,::-1,:,:] / (1.0 + ds.sphum.values[:,::-1,:,:])  # convert to mix-ratio

        if key == 'qc':
            dsout['qc'] = ds.clwmr.values[:,::-1,:,:]

        if key == 'qr':
            dsout['qr'] = ds.rwmr.values[:,::-1,:,:]

        if key == 'den':
            dsout['den'] = ds.delp.values[:,::-1,:,:]/(_grav*ds.delz.values[:,::-1,:,:])

        if key == 'rho':
            dsout['rho_star'] = ds.delp.values[:,::-1,:,:]/(_grav*ds.delz.values[:,::-1,:,:])  # simple way?
            
            qv                = ds.sphum.values[:,::-1,:,:] / (1.0 + ds.sphum.values[:,::-1,:,:])  # convert to mix-ratio  
            theta             = ds.theta.values[:,::-1,:,:]
            qcqr              = ds.clwmr.values[:,::-1,:,:] + ds.rwmr.values[:,::-1,:,:]
            dsout['rho']      = dsout['prs_dry'] / (_Rgas * theta * dsout['pii'] * (1.0 + _epsil*qv)*(1.0 - qcqr) )

            diff = dsout['rho_star'] - dsout['rho']
            print(f"--->RHO DIFF:  {diff.max()} {diff.min()}")

        if key == 'accum_prec':
            try:
                dsout['accum_prec'] = ds.rain_k.values[:,:,:]  # original name
            except:
                dsout['accum_prec'] = ds.accum_rain.values[:,:,:] # new name

        if key == 'div2d':
            print(" -->Computing finite difference 2D divergence\n")
            u = ds.ugrd.values[:,::-1,:,:]
            v = ds.vgrd.values[:,::-1,:,:]
            dudx = np.gradient(u, axis=3)
            dvdy = np.gradient(v, axis=2)
            dsout['div2d'] = dudx + dvdy

        if key == 'fft_div2d':

            print(" -->Computing FFT 2D divergence\n")

            x1 = ds.grid_xt.values
            y1 = ds.grid_yt.values
            Nx = x1.shape[0]
            Ny = y1.shape[0]

            if Nx != Ny:
                print("\n Warning, Nx != Ny, not sure if fft works!....\n")

            kx = np.fft.fftfreq(Nx) * 2*np.pi * Nx / (x1[-1] - x1[0])
            ky = np.fft.fftfreq(Ny) * 2*np.pi * Ny / (y1[-1] - y1[0])
            Kx, Ky  = np.meshgrid(kx, ky, indexing='ij')

            u  = ds.ugrd.values[:,::-1,:,:]
            v  = ds.vgrd.values[:,::-1,:,:]

            div2d = np.zeros_like(u)

            for n in np.arange(u.shape[0]):
                for k in np.arange(u.shape[1]):
                    div2d[n,k] = ifftn(1j * Kx * fftn(u[n,k])) \
                               + ifftn(1j * Ky * fftn(v[n,k]))

            dsout['fft_2d_div'] = np.real(div2d)

        if key == 'thetae':
            print(" -->Computing ThetaE \n")

            dsout['qv']   = ds.sphum.values[:,::-1,:,:] / (1.0 + ds.sphum.values[:,::-1,:,:])  # convert to mix-ratio
            dsout['temp'] = dsout['pii']*ds.theta.values[:,::-1,:,:]
            dsout['pres'] = dsout['prs_dry']

            dsout['thetae'] = compute_thetae(dsout)

        if key == 'total_e':
            
            print(" -->Computing Total energy \n")
            
            rv          = ds.sphum.values[:,::-1,:,:] / (1.0 + ds.sphum.values[:,::-1,:,:])  # convert to mix-ratio
            rl          = (ds.rwmr.values + ds.clwmr.values)[:,::-1,:,:]
            temperature = dsout['pii']*ds.theta.values[:,::-1,:,:]

            dens        = dsout['prs_dry'] / (_Rgas * temperature )

            dsout['total_e'] = dens * ( _Cv*temperature + _Cvv*rv*temperature + _Cpv*rl*temperature - _Lv*rl + _grav*(1.0+rv+rl)*dsout['zc'] )

        if key == 'theta_IC':
            nz = dsout['zc'].shape[1]
            ny = dsout['zc'].shape[2]
            nx = dsout['zc'].shape[3]

            with open(os.path.join(path, 'theta.bin'), 'rb') as f:
                dsout['theta_IC'] = np.fromfile(f, dtype=np.float32).reshape((nx, ny, nz), order='F').transpose()
        
            with open(os.path.join(path, 'qv.bin'), 'rb') as f:
                dsout['qv_IC'] = np.fromfile(f, dtype=np.float32).reshape((nx, ny, nz), order='F').transpose()

    if unit_test:
        write_Z_profile(dsout, model='SOLO', vars=variables, loc=(10,-1,1))
        
    if ret_dbz:   
        dsout['dbz']  = ds['reflectivity'].values[:,::-1,:,:]
        dsout['cref'] = dsout['dbz'].max(axis=1)

#       dsout = compute_dbz(dsout, version=2)
#       with open(dbz_filename, 'wb') as f:  np.save(f, dsout['dbz'])

    if ret_beta:   # Beta is a force, divide by density to get accel (density is based on Tv)

        # qv                = ds.sphum.values[0,::-1,0,0]  / (1.0 + ds.sphum.values[0,::-1,0,0])  # convert to mix-ratio  
        # den_base_state_1d = dsout['prs_dry'][0,::-1,0,0] / (_Rgas * ds.theta.values[0,::-1,0,0] * dsout['pii'][0,::-1,0,0] * (1.0 + _epsil * qv))
        # den_base_state_3D = np.broadcast_to(den_base_state_1d[np.newaxis, :, np.newaxis, np.newaxis], ds.delp.shape)

        # read in BETA
        
        print(f"\n Reading BETA from {os.path.join(path, 'w_b_accel.nc')}")

        dsbeta = xr.load_dataset(os.path.join(path, "w_b_accel.nc"), decode_times=False)

        # read forcing in as well.

        print(f"\n Reading density from {os.path.join(path, 'total_den.nc')}")

        dsrho = xr.load_dataset(os.path.join(path, "total_den.nc"), decode_times=False)

        den1d = dsrho.den.values[0,:,0,0]
        den3d = np.broadcast_to(den1d[np.newaxis, :, np.newaxis, np.newaxis], dsrho.den.shape)
 
        dsout['beta']  = dsbeta.beta.values / den3d
        dsout['rho_p'] = dsrho.den.values - den3d

        dsrho.close()
        dsbeta.close()
        
    print(f"\n Completed reading in:  {fpath}")
    print(f'-'*120)

    if zinterp is None:
        pass
    else:
        print(f"\n Interpolating fields to single column z-grid: {fpath} \n")

        new_shape = [dsout['zc'].shape[0],zinterp.shape[0],dsout['zc'].shape[2],dsout['zc'].shape[3],]

        for key in variables:
            if dsout[key].ndim == dsout['zc'].ndim:
                tmp =  interp_z(dsout[key], dsout['zc'], zinterp)
                dsout[key] = tmp

        if 'theta_IC' in variables:
            dsout['theta_IC'] = interp_z(dsout['theta_IC'], dsout['zc'][0], zinterp)
            dsout['qv_IC']    = interp_z(dsout['qv_IC'],    dsout['zc'][0], zinterp)

        dsout['zc'] = np.broadcast_to(zinterp[np.newaxis, :, np.newaxis, np.newaxis], new_shape)
        
        print(f" Finished interp fields to single column z-grid:  {path} \n")
        print(f'='*120)

    # if zinterp is None:
    #     pass
    # else:
    #     print(" Interpolating fields to single column z-grid:  %s \n" % path)

    #     new_shape = [dsout['zc'].shape[0], zinterp.shape[0], dsout['zc'].shape[2], dsout['zc'].shape[3],]

    #     for key in dsout:
    #         if dsout[key].ndim == dsout['zc'].ndim and key != 'zc':
    #             dsout[key] =  interp_z(dsout[key], dsout['zc'], zinterp)
                
    #     if ret_beta:
    #        dsout['beta'] = interp_z(dsout['beta'], dsout['zc'], zinterp)

    #     if ret_beta:
    #        dsout['beta'] = interp_z(dsout['beta'], dsout['zc'], zinterp)

    #     if 'theta_IC' in variables:
    #         dsout['theta_IC'] = interp_z(dsout['theta_IC'], dsout['zc'][0], zinterp)
    #         dsout['qv_IC']    = interp_z(dsout['qv_IC'],    dsout['zc'][0], zinterp)

    #     dsout['zc'] = np.broadcast_to(zinterp[np.newaxis, :, np.newaxis, np.newaxis], new_shape)

    #     print(" Finished interp fields to single column z-grid:  %s \n" % path)
    #     print(f'-'*120)

# Finish

    ds.close()

    if ret_obj:
        dsout = Dict2Object(dsout)

    if ret_ds:
        return dsout, ds
    else:
        return dsout

#########################################################################################
#
# Simple routine for fast read 
#

def read_solo_w(path, var='w', file_pattern=None, netCDF=False):

    from netCDF4 import MFDataset, Dataset

    print(f'-'*120)
    
    if file_pattern == None:
        
        # see if the path has the filename on the end....
        
        if os.path.basename(path)[:-3] != ".nc":
            fpath = os.path.join(path, _default_file)

    else:
       fpath = os.path.join(path, file_pattern)
            
    print(f" Now reading... {fpath}")
        
    try:
        fobj = Dataset(fpath)
    except:
        print(f"Cannot find the file in {fpath}, exiting")
        sys.exit(-1)
            
# storage bin

    dsout = {}

# Add two time variables

    dsout['sec'] = fobj.variables['time'][:]
    dsout['min'] = dsout['sec']/60.

# Always add coordinates to data structure

    dsout['xc'] = fobj.variables['grid_xt'][...] * 3000.
    dsout['yc'] = fobj.variables['grid_yt'][...] * 3000.

    ze = np.cumsum(fobj.variables['delz'][:,::-1,:,:], axis=1)

    dsout['ze']  = ze
    dsout['zc']  = np.zeros_like(ze)
    dsout['zc'][:,0,:,:]  = 0.5*ze[:,0,:,:]
    dsout['zc'][:,1:,:,:] = 0.5*(ze[:,:-1,:,:] + ze[:,1:,:,:])
    dsout['hgt'] = dsout['zc']

    if var == 'w': 
            dsout['w'] = fobj.variables['w'][:,::-1,:,:]

    print(f" Completed reading: {fpath} \n")
    
    return DictAsObject(dsout)

