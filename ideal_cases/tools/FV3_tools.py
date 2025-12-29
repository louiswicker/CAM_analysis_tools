import numpy as np
import netCDF4 as ncdf
import matplotlib.pyplot as plt
import xarray as xr
import glob as glob
import os as os
import glob
import sys as sys
from scipy.interpolate import CubicSpline
from para import fv3_spline_1d

from numpy.fft import fftn, ifftn, fftfreq

from timeit import default_timer as timer

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

def read_solo_fields(path, vars = [''], file_pattern=None, 
                     ret_ds=False, ret_obj=False, ret_beta=False, 
                     ret_pb=False, zinterp=None, unit_test=False):
        
    print(f'-'*120)

    start0 = timer()

    if file_pattern == None:
        
        # see if the path has the filename on the end....
        
        if os.path.basename(path)[:-3] != ".nc":
            fpath = os.path.join(path, _default_file)

    else:
       fpath = os.path.join(path, file_pattern)
            
    print(f" Now reading... {fpath}")
        
    start = timer()   # time the IO part
    try:
        ds = xr.open_dataset(fpath, decode_times=False)        
    except:
        print(f"Cannot find file in {fpath} exiting"  % path)
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

    start = timer()
        
# Add two time variables

    dsout['sec'] = ds.time.values[:]
    dsout['min'] = ds.time.values[:]/60.

# Always add coordinates to data structure

    dsout['xc'] = ds.grid_xt.values * 3000.
    dsout['yc'] = ds.grid_yt.values * 3000.

    ze = np.cumsum(ds.delz.values[:,::-1,:,:], axis=1)
    dz = ds.delz.values[:,::-1,:,:]

    dsout['zc']           = np.zeros_like(ze)
    dsout['zc'][:,0,:,:]  = 0.5*ze[:,0,:,:]
    dsout['zc'][:,1:,:,:] = 0.5*(ze[:,1:,:,:] + ze[:,:-1,:,:])
    dsout['hgt']          = dsout['zc']
    dsout['dz']           = dz

# Some variables we need a lot...the pressure calcs are pulled from L. Harris Jupyter notebook
    
    ptop              = ds.phalf[0]
    phalf             = ds.phalf[1:]
    pfull             = (ds.delp.cumsum(dim='pfull') + ptop).values
    dsout['prs_full'] = pfull[:,::-1,:,:]
    dsout['prs_dry']  = pfull[:,::-1,:,:]
    dsout['pii']      = (dsout['prs_dry'] / 100000.)**0.286

    dsout['ze']       = np.concatenate( (np.zeros((ze.shape[0], 1, ze.shape[2], ze.shape[3])), ze), axis=1)

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
            qcqr          = ds.clwmr.values[:,::-1,:,:] + ds.rwmr.values[:,::-1,:,:]
            dsout['buoy'] = _grav*(pert_th/base_th + 0.61*pert_qv - ds.clwmr.values[:,::-1,:,:] - ds.rwmr.values[:,::-1,:,:])
            dsout['dbuoy']= _grav*(pert_th/base_th + 0.61*pert_qv)
            
            den           = dsout['prs_dry'] / (_Rgas * theta * dsout['pii'] * (1.0 + _epsil*qv)*(1.0 - qcqr) )
            dsout['rbuoy'] = dsout['buoy'] * den

            variables = list(set(variables + ['dbuoy', 'rbuoy']))
            
        if key == 'dpstar':
            dsout['dpstar'] = ds.delp.values[:,::-1,:,:]

        if key == 'ppedge':
            try:
                tmp             = ds.wforcn[:,::-1,:,:].values
                dsout['ppedge'] = np.zeros((tmp.shape[0], tmp.shape[1]+1, tmp.shape[2], tmp.shape[3]))
                dsout['ppedge'][:,:-1,:,:] = tmp
            except:
                dsout['ppedge'] = np.zeros_like(ds.nhpres)

        if key == 'wrhs':
            ptop          = ds.phalf[0]*100.
            delp_m_qp     = ds.delp * (1.0 - ds.qp)
            peg           = np.zeros((ds.dims['time'],ds.dims['phalf'],ds.dims['grid_yt'],ds.dims['grid_xt']),)
            peg[:,0,:,:]  = ptop
            peg[:,1:,:,:] = (delp_m_qp.cumsum(dim='pfull')+ptop).values
            peglog        = np.log(peg)
            pm2           = ((peg[:,1:,:,:] - peg[:,:-1,:,:]) / (peglog[:,1:,:,:] - peglog[:,:-1,:,:]))[:,::-1,:,:]

            pNH_interp    = np.exp(interp_z(np.log(ds.nhpres.values[:,::-1,:,:]), dsout['zc'], zinterp))
            pm2_interp    = np.exp(interp_z(np.log(pm2), dsout['zc'], zinterp))
            rNH_1D        = pm2_interp[0,:,0,0]
            pert_p        = pNH_interp - rNH_1D[None, :, None, None]

            dm2           = interp_z(delp_m_qp.values[:,::-1,:,:], dsout['zc'], zinterp) / _grav
            dz            = interp_z(dsout['ze'][:,1:,:,:] - dsout['ze'][:,:-1,:,:], dsout['zc'], zinterp)

            pe            = np.zeros((ds.dims['time'], dm2.shape[1]+1, ds.dims['grid_yt'], ds.dims['grid_xt']))

            print(" Checking shape of Pe\n", pe.shape)

            # Use FV3 solver fortran code to obtain pert nh-pressure on grid edges (fv3_spline_1d)
            print(" Computing pert nh-pressure on grid edges\n")
            
            for t in np.arange(dsout['zc'].shape[0]):
                # if t % 5 == 0: print(t)
                for j in np.arange(dsout['zc'].shape[2]):
                    for i in np.arange(dsout['zc'].shape[3]):

                        pe[t,:,j,i] = fv3_spline_1d(pert_p[t,:,j,i], dm2[t,:,j,i], pert_p.shape[1])

            dp             = pe[:,1:,:,:] - pe[:,:-1,:,:] # compute pressure difference across cell
            dsout['wrhs']  = ( -  dp / dm2)
            dsout['wrhs2'] = ( -  dp / dm2)
            
            dsout['wrhs_dp'] = pe[:,:-1,:,:]
            dsout['wrhs_dm'] = dm2

        if key == 'u': 
            dsout['u'] = ds.ucomp.values[:,::-1,:,:]

        if key == 'v': 
            dsout['v'] = ds.vcomp.values[:,::-1,:,:]

        if key == 'w': 
            dsout['w'] = ds.w.values[:,::-1,:,:]

        if key == 'dzdt': 
            dsout['dzdt'] = 0.5*(ds.dzdt.values[:,1:,:,:] + ds.dzdt.values[:,:-1,:,:])[:,::-1,:,:]

        if key == 'vvort': 
            dsout['vvort'] = ds.rel_vort.values[:,::-1,:,:]

        if key == 'pres':
            dsout['pres'] = ds.nhpres.values[:,::-1,:,:]

        if key == 'pert_p':    # code from L Harris Jupyter notebook
            ptop       = ds.phalf[0]*100.
            delp_m_qp  = ds.delp * (1.0 - ds.clwmr - ds.rwmr)
            #delp_m_qp  = ds.delp 

            peg           = np.zeros((ds.dims['time'],ds.dims['phalf'],ds.dims['grid_yt'],ds.dims['grid_xt']),)
            peg[:,0,:,:]  = ptop
            peg[:,1:,:,:] = (delp_m_qp.cumsum(dim='pfull')+ptop).values
            peglog        = np.log(peg)
            pm2           = ((peg[:,1:,:,:] - peg[:,:-1,:,:]) / (peglog[:,1:,:,:] - peglog[:,:-1,:,:]))[:,::-1,:,:]
                        
            p_from_qv  = interp_z(((ds.sphum)*ds.delp).cumsum(dim='pfull').values[:,::-1,:,:], dsout['zc'], zinterp )            
            p_from_qp  = ds.qp.cumsum(dim='pfull') # already * dpstar
            p_from_qn  = ds.qn.cumsum(dim='pfull') # already * dpstar
            # p_from_qp  = interp_z((ds.clwmr*ds.delp).cumsum(dim='pfull').values[:,::-1,:,:], dsout['zc'], zinterp )  # already * dpstar
            # p_from_qn  = interp_z((ds.rwmr*ds.delp).cumsum(dim='pfull').values[:,::-1,:,:], dsout['zc'], zinterp )  # already * dpstar

            p_from_liq = p_from_qp.values[:,::-1,:,:] # + p_from_qn

            pNH_interp       = np.exp(interp_z(np.log(ds.nhpres.values[:,::-1,:,:]), dsout['zc'], zinterp))
            pm2_interp       = np.exp(interp_z(np.log(pm2), dsout['zc'], zinterp))
            rNH_1D           = pm2_interp[0,:,0,0]
            dsout['pert_p2'] = pNH_interp - rNH_1D[None, :, None, None]
            rNH_1D           = pNH_interp[0,:,0,0]
            dsout['pert_p3'] = pNH_interp - rNH_1D[None, :, None, None] 
            dsout['pert_p4']  = np.exp(interp_z(np.log(ds.nhpres_pert.values[:,::-1,:,:]), dsout['zc'], zinterp))
            dsout['pert_p']   = ds.nhpres_pert.values[:,::-1,:,:]

            # remove mean
            psum = dsout['pert_p2'].sum(axis=(1,2,3))
           # print(" \n Mean PERT_P2 pressure before: ", psum)
            dsout['pert_p2'] =  dsout['pert_p2'] - psum[:,None,None,None] * psum.size / dsout['pert_p2'].size
            psum = dsout['pert_p2'].sum(axis=(1,2,3))
            # print(" \n Mean PERT_P2 pressure after: ", psum)

            psum = dsout['pert_p3'].sum(axis=(1,2,3))
            # print(" \n Mean PERT_P3 pressure before: ", psum)
            dsout['pert_p3'] =  dsout['pert_p3'] - psum[:,None,None,None] * psum.size / dsout['pert_p3'].size
            psum = dsout['pert_p2'].sum(axis=(1,2,3))
            # print(" \n Mean PERT_P3 pressure after: ", psum)

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

        if key == 'rwqv':
            delp          = (ds.delp * (1.0 - ds.clwmr - ds.rwmr)).values[:,::-1,:,:]
            den           = delp / (_grav*ds.delz.values[:,::-1,:,:])
            qv            = ds.sphum.values[:,::-1,:,:] / (1.0 + ds.sphum.values[:,::-1,:,:])
            dsout['rwqv'] = den * ds.w.values[:,::-1,:,:] * qv
         
        if key == 'rw':
            dsout['rw']   = den * ds.w.values[:,::-1,:,:]

        if key == 'total_q':
            delp          = (ds.delp * (1.0 - ds.clwmr - ds.rwmr)).values[:,::-1,:,:]
            den           = delp / (_grav*ds.delz.values[:,::-1,:,:])
            qv            = ds.sphum.values[:,::-1,:,:] / (1.0 + ds.sphum.values[:,::-1,:,:])
            dsout['total_q'] = den * (qv + ds.clwmr.values[:,::-1,:,:] + ds.rwmr.values[:,::-1,:,:])
        
        if key == 'rho':
            dsout['rho_star'] = ds.delp.values[:,::-1,:,:]/(_grav*ds.delz.values[:,::-1,:,:])  # simple way?
            
            qv                = ds.sphum.values[:,::-1,:,:] / (1.0 + ds.sphum.values[:,::-1,:,:])  # convert to mix-ratio  
            theta             = ds.theta.values[:,::-1,:,:]
            qcqr              = ds.clwmr.values[:,::-1,:,:] + ds.rwmr.values[:,::-1,:,:]
            dsout['rho']      = dsout['prs_dry'] / (_Rgas * theta * dsout['pii'] * (1.0 + _epsil*qv)*(1.0 - qcqr) )

            diff = dsout['rho_star'] - dsout['rho']
            print(f"--->RHO DIFF:  {diff.max()} {diff.min()}")

            variables = list(set(variables + ['rho_star']) )
            
        if key == 'accum_prec':
            try:
                dsout['accum_prec'] = ds.rain_k.values[:,:,:]  # original name
            except:
                dsout['accum_prec'] = ds.accum_rain.values[:,:,:] # new name

        if key == 'div2d':
            print(" -->Computing finite difference 2D divergence\n")
            u = ds.ucomp.values[:,::-1,:,:]
            v = ds.vcomp.values[:,::-1,:,:]
            dudx = np.gradient(u, axis=3)
            dvdy = np.gradient(v, axis=2)
            dsout['div2d'] = dudx + dvdy / (dsout['xc'][1] - dsout['xc'][0])

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

            variables = list(set(variables + ['theta_IC','qv_IC']) )

        if key == 'dbz':
            dsout['dbz']  = ds['reflectivity'].values[:,::-1,:,:]
            dsout['cref'] = dsout['dbz'].max(axis=1)

    if unit_test:
        write_Z_profile(dsout, model='SOLO', vars=variables, loc=(10,-1,1))
        

#       dsout = compute_dbz(dsout, version=2)
#       with open(dbz_filename, 'wb') as f:  np.save(f, dsout['dbz'])

    if ret_beta:   # Beta is a force, divide by density to get accel (density is based on Tv)

        # qv                = ds.sphum.values[0,::-1,0,0]  / (1.0 + ds.sphum.values[0,::-1,0,0])  # convert to mix-ratio  
        # den_base_state_1d = dsout['prs_dry'][0,::-1,0,0] / (_Rgas * ds.theta.values[0,::-1,0,0] * dsout['pii'][0,::-1,0,0] * (1.0 + _epsil * qv))
        # den_base_state_3D = np.broadcast_to(den_base_state_1d[np.newaxis, :, np.newaxis, np.newaxis], ds.delp.shape)

        # read in BETA
        
        print(f"\n Reading BETA from {os.path.join(path, 'beta.nc')}")

        dsbeta = xr.load_dataset(os.path.join(path, "beta.nc"), decode_times=False)

        # read forcing in as well.

        print(f"\n Reading density from {os.path.join(path, 'den.nc')}")

        dsrho = xr.load_dataset(os.path.join(path, "den.nc"), decode_times=False)

        den1d = dsrho.den.values[0,:,0,0]
        den3d = np.broadcast_to(den1d[np.newaxis, :, np.newaxis, np.newaxis], dsrho.den.shape)
 
        dsout['beta']  = dsbeta.beta.values / den3d
        dsout['rho_p'] = dsrho.den.values - den3d

        dsrho.close()
        dsbeta.close()

    if ret_pb:   

        # read in derived pressure buoyancy
        
        print(f"\n Reading Pb from {os.path.join(path, 'pb.nc')}")

        dspb = xr.load_dataset(os.path.join(path, "pb.nc"), decode_times=False)

        # for each time, set mean to zero

        dsout['pb']  = dspb.Pb.values
        psum         = dsout['pb'].sum(axis=(1,2,3))
        # print(" \n Mean pressue before: ", psum)

        dsout['pb'] =  dsout['pb'] - psum[:,None,None,None] * psum.size / dsout['pb'].size

        psum         = dsout['pb'].sum(axis=(1,2,3))
        # print(" \n Mean pressue after: ", psum)
        
        dspb.close()
  
    print(f"\n Completed reading in:  {fpath} --> time to process variables: {timer() - start:.2f} sec ")

    if zinterp is None:
        pass
    else:
        print(f"\n Interpolating fields to single column z-grid: {fpath} \n")
        
        start = timer()

        new_shape = [dsout['zc'].shape[0],zinterp.shape[0],dsout['zc'].shape[2],dsout['zc'].shape[3]]

        for key in variables:
             if dsout[key].ndim == dsout['zc'].ndim:
                try:
                    tmp =  interp_z(dsout[key], dsout['zc'], zinterp)
                    dsout[key] = tmp
                except ValueError:
                    print(f" Interpolation could not be done on {key}, as shape is {dsout[key].shape}")

        dsout['zc'] = np.broadcast_to(zinterp[np.newaxis, :, np.newaxis, np.newaxis], new_shape)

        print(f" Interpolated completed, Total CPU:  {path}: {timer()- start:.2f} sec \n")  

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
        fobj = Dataset(fpath).variables
    except:
        print(f"Cannot find the file in {fpath}, exiting")
        sys.exit(-1)
            
# storage bin

    dsout = {}

# Add two time variables

    dsout['sec'] = fobj['time'][:]
    dsout['min'] = dsout['sec']/60.

# Always add coordinates to data structure

    dsout['xc'] = fobj['grid_xt'][...] * 3000.
    dsout['yc'] = fobj['grid_yt'][...] * 3000.

    ze = np.cumsum(fobj['delz'][:,::-1,:,:], axis=1)

    dsout['ze']  = ze
    dsout['zc']  = np.zeros_like(ze)
    dsout['zc'][:,0,:,:]  = 0.5*ze[:,0,:,:]
    dsout['zc'][:,1:,:,:] = 0.5*(ze[:,:-1,:,:] + ze[:,1:,:,:])
    dsout['hgt'] = dsout['zc']

    if var == 'w': 
            dsout['w'] = fobj['w'][:,::-1,:,:]

    print(f" Completed reading: {fpath} \n")
    
    return DictAsObject(dsout)

def read_solo_w2(path, var='w', file_pattern=None, netCDF=False):

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
        fobj = xr.load_dataset(fpath, chunks="auto", decode_times=False)
    except:
        print(f"Cannot find the file in {fpath}, exiting")
        sys.exit(-1)
            
# storage bin

    dsout = {}

# Add two time variables

    dsout['sec'] = fobj['time'][:]
    dsout['min'] = dsout['sec']/60.

# Always add coordinates to data structure

    dsout['xc'] = fobj['grid_xt'][...] * 3000.
    dsout['yc'] = fobj['grid_yt'][...] * 3000.

    ze = np.cumsum(fobj.delz.values[:,::-1,:,:], axis=1)
    dz = fobj.delz.values[:,::-1,:,:]

    dsout['zc']           = np.zeros_like(ze)
    dsout['zc'][:,0,:,:]  = 0.5*ze[:,0,:,:]
    dsout['zc'][:,1:,:,:] = 0.5*(ze[:,1:,:,:] + ze[:,:-1,:,:])
    dsout['hgt'] = dsout['zc']
    dsout['dz']  = dz

    if var == 'w': 
            dsout['w'] = fobj['w'][:,::-1,:,:]

    print(f" Completed reading: {fpath} \n")
    
    return DictAsObject(dsout)

