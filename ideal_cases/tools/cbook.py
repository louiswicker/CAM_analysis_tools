import numpy as np
import xarray as xr
import glob as glob
import os as os
import glob
import sys as sys
import pickle
import time

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

# GFDL MP constants
_rhor  = 1.0e3
_vconr = 2503.23638966667
_normr = 25132741228.7183
_qmin  = 1.0e-12

#--------------------------------------------------------------------------------------------------
# Code from Joel McClune to convert dictionaries to objects

class DictObj:
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
               setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
               setattr(self, key, DictObj(val) if isinstance(val, dict) else val)

#----------------------------------------------------------------------------
#

def pickle2Obj(file, retObj=True):
    """
        Code to read a dictionary written to a pickle file, and if needed convert to a object
    """
    with open(file, 'rb') as f:
        file = pickle.load(f)
        f.close()
        if retObj == True:
            return(DictObj(file))
        else:
            return file
#
#--------------------------------------------------------------------------------------------------

def get_percentile_value(field, percentile=0.99):
    
    flat_fld = field.flatten()

    sorted = np.sort(flat_fld)

    idx = int(percentile * flat_fld.shape[0])

    print("\n Percentile:  %f  Index:  %d   Value:  %f" % (percentile, idx, sorted[idx]) )
    
    return sorted[idx]
#----------------------------------------------------------------------------
#

def compute_dbz(state, version=2):

    """
        version == 1:
        Uses Thompson microphysics code adapted by L. Reames at NSSL.
        
        version == 2:
        The Marshall-Palmer code is adopted from the GFDL Solo rad_rar.F90
        module.  Copyright is owned by GFDL and SJ Lin. Thanks for sharing.
        

        # sjl notes: marshall - palmer, dbz = 200 * precip ** 1.6, 
        #            precip = 3.6e6 * t1 / rhor * vtr ! [mm / hr]
        #            gfdl_mp terminal fall speeds are used
        #            account for excessively high cloud water - > autoconvert (diag only) excess cloud water
        # date last modified 20170701
    """

    temp   = state['temp']
    pres   = state['pres']
    qv     = state['qv']
    qc     = state['qc']
    qr     = state['qr']

#     if version < 2:  # use thompson fortran code
    
#         print("\n Running Thompson radar reflectivity code")
        
#         qi     = np.zeros_like(qv)
#         qs     = np.zeros_like(qv)
#         qg     = np.zeros_like(qv)
#         dbz    = np.zeros_like(qv)

#         try:
#             from cmpref import cmpref_mod as cmpref
#         except:
#             print("\n Cannot load Thompson microphysics fortran module")
#             state['dbz'] = -35.0
#             return state
            
#         for n in np.arange(temp.shape[0]):
#             dbz[n] = cmpref.calcrefl10cm(qv[n], qc[n], qr[n], qs[n], qg[n], temp[n], pres[n])
#             print(n, dbz[n].max(), dbz[n].min())
            
#     else:  # use GFLD Solo Marshall-Palmer code....
    
    print("\n Running Marshall-Palmer radar reflectivity code")

    den    = pres / (temp*_Rgas)
    denfac = 1.2 / den

    denfac = np.sqrt( np.where(denfac < 10.0, denfac, 10.) )
    qcmin  = np.where(qc - 1.0e-3 > 0.0, qc-1.0e-3, 0.0)
    dbz    = -35.0 * np.ones_like(qv)
    t1     = qr + qcmin
    t1     = den * np.where(t1 > _qmin, t1, _qmin )
    vtr    = _vconr * denfac * (t1 / _normr)**0.2
    vtr    = np.where(vtr > 1.e-3, vtr, 1.0e-3)
    z_e    = 200. * (3.6e6 * (t1 / _rhor) * vtr)**1.6
    dbz    = 10. * np.log10 (np.where(z_e > 0.01, z_e, 0.1))
        
#         for n in np.arange(temp.shape[0]):
#             print(n, dbz[n].max(), dbz[n].min())
        
    state['dbz'] = dbz
    
    return state

#
#--------------------------------------------------------------------------------------------------
def interp_z(data, zin, zout):
    """
    """

    debug = False

    start = time.time()

    ndim = data.ndim
    cdim = zin.ndim

    if debug:  
        print("\n Data shape: ", data.shape, " Z shape:  ", zin.shape)

    if ndim == 1:
        dinterp = np.interp(zout, zin, data)

    if ndim == 2:

        if cdim == 1:
            zND = np.broadcast_to(zin[:,np.newaxis], data.shape)
        elif cdim == ndim:
            zND = zin
        else:
            print("--> INTERP_Z input z array must be 1D or same DIM as data array\n")
            return None

        dinterp = np.zeros((len(zout),data.shape[1]),dtype=np.float32)

        for t in np.arange(data.shape[1]):
            dinterp[:,t] = np.interp(zout, zND[:,t], data[:,t])
        if debug:  print(dinterp.shape)

    if ndim == 3:

        if cdim == 1:
            zND = np.broadcast_to(zin[:, np.newaxis, np.newaxis], data.shape)
        elif cdim == ndim:
            zND = zin
        else:
            print("--> INTERP_Z input z array must be 1D or same DIM as data array\n")

        dinterp = np.zeros((len(zout),data.shape[1],data.shape[2]),dtype=np.float32)

        for i in np.arange(data.shape[2]):
            for j in np.arange(data.shape[1]):
                dinterp[:,j,i] = np.interp(zout, zND[:,j,i], data[:,j,i])

    if ndim == 4:

        if cdim == 1:
            zND = np.broadcast_to(zin[np.newaxis, :, np.newaxis, np.newaxis], data.shape)
        elif cdim == ndim:
            zND = zin
        else:
            print("--> INTERP_Z input z array must be 1D or same DIM as data array\n")

        dinterp = np.zeros((data.shape[0],len(zout),data.shape[2],data.shape[3]),dtype=np.float32)

        for t in np.arange(data.shape[0]):
            for j in np.arange(data.shape[2]):
                for i in np.arange(data.shape[3]):
                    dinterp[t,:,j,i] = np.interp(zout, zND[t,:,j,i], data[t,:,j,i])

    if debug:  print("\n Total time taken for interpolation: ", time.time() - start)

    return dinterp
        
        
#--------------------------------------------------------------------------------------------------
#

def write_Z_profile(state, model='WRF', loc=(0,-1,-1), vars = ['hgt', 'press', 'theta', 'den', 'pert_p']):

    try:
        from columnar import columnar
    except:
        print("\n-----------------------------------------------\n")
        print("Missing package columnar for printing...., exiting")
        print("\n-----------------------------------------------\n")
        return

    #----------------------------------------------------------------------------
    #

    def print_kernel(state, loc, subvars):

        newlist = []
        newlist.append(np.arange(state[vars[0]].shape[1]).tolist())
        headers = ['level']

        t_idx = loc[0]
        i_idx = loc[1]
        j_idx = loc[2]

        for key in subvars:
        
            ds = state[key]

            if ds.ndim < 4:
                print("\n Variable %s does not have 4 dims, skipping...\n" % key)
                continue
        
            if t_idx < 0:
                ds = ds.mean(axis=0)
            else:
                ds = ds[t_idx]

            if i_idx < 0 or j_idx < 0:
                ds = ds.mean(axis = (1,2))
            else:
                ds = ds[:,:,j_idx,i_idx]
 
            newlist.append(ds.tolist())
            headers.append(key)

        # need to rearange all the state into row like an excel spreadsheet
    
        row_data = [list(x) for x in zip(*newlist)]
        
        table = columnar(row_data, headers, no_borders=True)
        print(table)

    #
    #----------------------------------------------------------------------------
    
    print('#-----------------------------#')
    print('          %s                  ' % model)
    print('#-----------------------------#')
    
    # if the number of variables to be print is large - split the printing up...

    for i in range(0, len(vars), 5):

        subvars = vars[i:i + 5]

        print("\n Now printing: ",subvars)
        
        print_kernel(state, loc, subvars)

    return
    
#
#--------------------------------------------------------------------------------------------------

    
    
    
    
