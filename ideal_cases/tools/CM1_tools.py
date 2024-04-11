import numpy as np
import xarray as xr
import glob as glob
import os as os
import glob
import sys as sys

from numpy.fft import fft2, ifft2, fftfreq

from tools.cbook import write_Z_profile, compute_dbz, compute_thetae

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
                   'pres',   
                   'pert_p',   
                   'pii',     
                   'accum_prec',
                   'theta',    
                   'pert_th',   
                   'thetae',
                   'buoy',
                   'dwdt',
                   ]

#---------------------------------------------------------------------
def interp_z(data, zin, zout):

    import time

    debug = True

    start = time.time()

    ndim = data.ndim
    cdim = zin.ndim

    if debug:  print('\n',ndim, cdim)

    if ndim == 1:
        dinterp = np.interp(zout, zin, data)

    if ndim == 2:

        dinterp = np.zeros((data.shape[0],len(zout)),dtype=np.float32)

        if cdim < ndim:
            z2d = np.broadcast_to(zin[np.newaxis, :], data.shape)

        for t in np.arange(data.shape[0]):
            dinterp[t,:] = np.interp(zout, z2d, data[t,:])

    if ndim == 3: 

        dinterp = np.zeros((len(zout),data.shape[1],data.shape[2]),dtype=np.float32)

        if cdim < ndim:
            z3d = np.broadcast_to(zin[:, np.newaxis, np.newaxis], data.shape)

        for i in np.arange(data.shape[2]):
            for j in np.arange(data.shape[1]):
                dinterp[:,j,i] = np.interp(zout, z3d[:,j,i], data[:,j,i])

    if ndim == 4:

        dinterp = np.zeros((data.shape[0],len(zout),data.shape[2],data.shape[3]),dtype=np.float32)

        if cdim < ndim:
            z4d = np.broadcast_to(zin[np.newaxis, :, np.newaxis, np.newaxis], data.shape)

        for t in np.arange(data.shape[0]):
            for j in np.arange(data.shape[2]):
                for i in np.arange(data.shape[3]):
                    dinterp[t,:,j,i] = np.interp(zout, z4d[t,:,j,i], data[t,:,j,i])

    if debug:  print("\n Total time taken for interpolation: ", time.time() - start) 

    
    return dinterp
                   
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

# Add two time variables

    dsout['sec'] = ds.time.values[:]
    dsout['min'] = ds.time.values[:]/60.

# Add some spatial info

    dsout['xc'] = ds.xh.values
    dsout['yc'] = ds.yh.values
    dsout['zc'] = np.broadcast_to(1000.*ds.zh.values[np.newaxis, :, np.newaxis, np.newaxis], ds.prs.shape)
    dsout['ze'] = np.broadcast_to(1000.*ds.zf.values[np.newaxis, :, np.newaxis, np.newaxis], ds.w.shape)

    for key in variables:

        if key == 'theta': 
            dsout['theta'] = ds.th.values
            
        if key == 'pert_th': 
            dsout['pert_th'] = ds.thpert.values
            
        if key == 'buoy': 
            qv0 = ds.qv[0,:,-1,-1].values
            qv0 = np.broadcast_to(qv0[np.newaxis, :, np.newaxis, np.newaxis], ds.qv.shape)
            dsout['buoy'] = 9.806*(ds.thpert.values/ds.th0.values + 0.61*(ds.qv.values-qv0) \
                          - ds.qc.values - ds.qr.values)
        if key == 'pert_t':
            base_pii = ds.pi0.values  
            base_th0 = ds.th0.values
            base_t0  = base_th0*base_pii
            pii      = (ds.prs.values / 100000.)**0.286
            temp     = ds.th.values * pii
            dsout['pert_t'] = temp - base_t0
 
        if key == 'u': 
            dsout['u'] = 0.5*(ds.u[:,:,:,1:] + ds.u[:,:,:,:-1]).values

        if key == 'v': 
            dsout['v'] = 0.5*(ds.v[:,:,1:,:] + ds.v[:,:,:-1,:]).values

        if key == 'w': 
            try:
                dsout['w'] = ds.winterp.values
            except:
                dsout['w'] = 0.5*(ds.w[:,1:,:,:] + w[:,:-1,:,:]).values

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
            pii  = (ds.prs.values / 100000.)**0.286
            dsout['den'] = ds.prs.values / (_Rgas*ds.th.values*pii)
            dsout['row'] = ds.prs.values / (_Rgas*ds.th.values*pii)
            
        if key == 'row':
            pii  = (ds.prs.values / 100000.)**0.286
            dsout['den'] = ds.prs.values / (_Rgas*ds.th.values*pii)
            
        if key == 'temp':
            pii  = (ds.prs.values / 100000.)**0.286
            dsout['temp'] = ds.th.values*pii

        if key == 'pii':
            dsout['pii'] = (ds.prs.values / 100000.)**0.286

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

        if key == 'thetae':
            print(" -->Computing ThetaE \n")
            
            dsout['qv']   = ds.qv.values 
            pii           = (ds.prs.values / 100000.)**0.286
            dsout['temp'] = ds.th.values*pii
            dsout['pres'] = ds.prs.values

            dsout['thetae'] = compute_thetae(dsout)

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
 
