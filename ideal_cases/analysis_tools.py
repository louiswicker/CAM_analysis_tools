import numpy as np
import netCDF4 as ncdf
import matplotlib.pyplot as plt
import xarray as xr
import glob as glob
import os as os
import glob
import sys as sys
import matplotlib as mpl

import tools
from tools.FV3_tools import read_solo_fields
from tools.CM1_tools import read_cm1_fields
from tools.WRF_tools import read_wrf_fields
from tools.thermo import compute_thetae
from tools.cbook import get_percentile_value

import warnings
warnings.filterwarnings("ignore")

# def fxn():
#     warnings.warn("deprecated", DeprecationWarning)

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     fxn()
    
_nthreads = 2
_Rgas     = 287.04
_gravity  = 9.806

#--------------------------------------------------------------------------------------------------
# Interp from 3D pressure to 1D pressure (convert from hybrid to constant p-levels)

def interp3d_np(data, p3d, p1d, nthreads = _nthreads):
    
    dinterp = np.zeros((len(p1d),data.shape[1],data.shape[2]),dtype=np.float32)

    if nthreads < 0:  # turning this off for now.
        def worker(i):
            print("running %d %s" % (i, data.shape))
            for j in np.arange(data.shape[1]):
                  dinterp[:,j,i] = np.interp(p1d[::-1], p3d[:,j,i], data[:,j,i])

        pool = mp.Pool(nthreads)
        for i in np.arange(data.shape[2]):
            pool.apply_async(worker, args = (i, ))
        pool.close()
        pool.join()
        
        return dinterp[::-1,:,:]
    
    else:        
        for i in np.arange(data.shape[2]):
            for j in np.arange(data.shape[1]):
                dinterp[:,j,i] = np.interp(p1d[::-1], p3d[:,j,i], data[:,j,i])
        
        return dinterp[::-1,:,:]

#--------------------------------------------------------------------------------------------------
#
# GENERATE IDEAL PROFILES
#

def generate_ideal_profiles(path, model_type='wrf', file_pattern=None,
                            w_thresh = 5.0, cref_thresh = 35., 
                            min_pix = 3, percentile=None, **kwargs):
    
    print(f'-'*120,'\n')
    print(" Processing model run:  %s \n" % path)
    
    if percentile:
        print(" Processing objects with CREF percentile:  %f \n" % percentile)
    
    if model_type == 'fv3_solo' or model_type == 'solo':
    
        ds = read_solo_fields(path, file_pattern=file_pattern,
                              vars=['hgt', 'pres', 'w', 'temp', 'theta', 'pert_t', 'pert_th',
                                    'qv', 'pert_p', 'pert_pss'], ret_dbz=True)
        
        ds['thetae'] = compute_thetae(ds)
        
        if percentile:
            cref_thresh = get_percentile_value(ds['dbz'].max(axis=1), percentile=percentile)
            print("\n SOLO/FV3 CREF percentile value:  %f" % cref_thresh)

        profiles = compute_obj_profiles(ds,
                                        w_thresh = w_thresh, cref_thresh = cref_thresh, 
                                        min_pix=min_pix, 
                                        extra_vars=['temp', 'theta', 'thetae', 'pert_t', 'pert_th',
                                                    'qv', 'pert_p', 'pert_pss'], **kwargs)

        return profiles
    
    if model_type == 'wrf':
        
        ds = read_wrf_fields(path, file_pattern=file_pattern,
                             vars=['hgt', 'pres', 'w', 'temp', 'theta', 'pert_t', 'pert_th',
                                   'qv', 'pert_p'], ret_dbz=True)
                                                   
        ds['thetae'] = compute_thetae(ds)

        if percentile:
            cref_thresh = get_percentile_value(ds['dbz'].max(axis=1), percentile=percentile)
            print("\n WRF CREF percentile value:  %f" % cref_thresh)

        profiles = compute_obj_profiles(ds,
                                        w_thresh = w_thresh, cref_thresh = cref_thresh, 
                                        min_pix=min_pix,  
                                        extra_vars=['temp', 'theta', 'thetae', 'pert_t', 'pert_th',
                                                     'qv', 'pert_p'], **kwargs)
        
        return profiles
        
    if model_type == 'cm1':
        
        ds = read_cm1_fields(path, file_pattern=file_pattern,
                              vars=['hgt', 'pres', 'w', 'temp', 'theta', 'pert_t', 'pert_th',
                                    'qv', 'pert_p'], ret_dbz=True)

        ds['thetae'] = compute_thetae(ds)
        
        if percentile:
            cref_thresh = get_percentile_value(ds['dbz'].max(axis=1), percentile=percentile)
            print("\n CM1 CREF percentile value:  %f" % cref_thresh)
                                                    
        profiles = compute_obj_profiles(ds,
                                        w_thresh = w_thresh, cref_thresh = cref_thresh, 
                                        min_pix=min_pix, 
                                        extra_vars=['temp', 'theta', 'thetae', 'pert_t', 'pert_th',
                                                    'qv', 'pert_p'], **kwargs)

        return profiles

#-------------------------------------------------------------------------------
def compute_obj_profiles(ds, w_thresh = 3.0, cref_thresh = 45., min_pix=5, 
                         zhgts = 250. + 250.*np.arange(100), extra_vars = None):
    
    from skimage.measure import label

    #---------------------------------------------------------------------
    # local interp function
    def interp3dz_np(data, z3d, z1d, nthreads = _nthreads):

        dinterp = np.zeros((len(z1d),data.shape[1]),dtype=np.float32)

        if nthreads < 0:  # turning this off for now.
            def worker(j):
                print("running %d %s" % (i, data.shape))
                dinterp[:,j] = np.interp(z1d, z3d[:,j], data[:,j])

            pool = mp.Pool(nthreads)
            for i in np.arange(data.shape[2]):
                pool.apply_async(worker, args = (i, ))
            pool.close()
            pool.join()

            return dinterp

        else:        
            for j in np.arange(data.shape[1]):
                dinterp[:,j] = np.interp(z1d, z3d[:,j], data[:,j])

            return dinterp
        
    # check to see if we have the variable keys we need for processing...
    
    if all(key in ds for key in ['w', 'dbz', 'hgt', 'pres']):
        print("\n COMPUTE_OBJ_PROFILES: found needed object variables")
    else:
        for key in ['w', 'dbz', 'hgt', 'pres']:
            try:
                dummy = ds[key]
            except:
                print("\n COMPUTE_OBJ_PROFILES: PROBLEM - Input data dictionary is missing key:  %s" % key)
        return None
   
    # thanks to Larissa Reames for suggestions on how to do this (and figuring out what to do!)
    
    mask_cref   = np.where(ds['dbz'].max(axis=1) > cref_thresh, True, False)
    mask_w_3d   = np.where(ds['pres'] < 70000.0, ds['w'], np.nan)
    
    mask_w_2d   = np.nanmax(mask_w_3d, axis=1)
    mask_w_cref = (mask_w_2d > w_thresh) & mask_cref
    f_mask      = mask_w_cref.astype(np.int8)
        
    tindex   = [0]
    wlist    = []
    slist    = []
    
    vars         = {}
    vars['dbz']  = ds['dbz']
    vars['w']    = ds['w']
    
    p_vars       = {}   # this is for the profiles we will create
    p_vars['dbz']= []
    p_vars['w']  = []
    
    if extra_vars != None:
        for key in extra_vars:
            vars[key]   = ds[key]  # this should be an array same shape as W, PRES, DBZ
            p_vars[key] = []
                
    all_obj  = 0
    w_obj    = 0
    
    for n in np.arange(vars['w'].shape[0]): # loop over number of time steps.            
        
        # check to see if there are objects
        
        if (np.sum(f_mask[n]) == 0):
            
            tindex.append(w_obj)
                            
            continue

        else:
            
            # returns a 2D array of labels for updrafts)
            label_array, num_obj = label(f_mask[n], background=0, connectivity=2, return_num = True) 
            
            all_obj += num_obj

            if( num_obj > 0 ):                                     # if there is more than the background object, process array.

                for l in np.arange(1,num_obj):                     # this is just a list of 1,2,3,4.....23,24,25....
                    npix = np.sum(label_array == l)                # this is a size check - number of pixels assocated with a label
                    if npix >= min_pix:
                        jloc, iloc = np.where(label_array == l)    # extract out the locations of the updrafts 
                        w_obj += 1
                        if len(iloc) > 0 and len(jloc) > 0:
                           
                            zraw    = ds['hgt'][n,:,jloc,iloc]             # get z_raw profiles
                            
                            for key in p_vars:                     # loop through dictionary of variables.
                                tmp = vars[key][n,:,jloc,iloc]
                                profile = interp3dz_np(tmp.transpose(), zraw.transpose(), zhgts, nthreads = _nthreads)
                                p_vars[key].append([profile.mean(axis=(1,))],)

                                if key == 'w':
                                    slist.append(tmp.shape[0])
                            
                                # returns columns of variables interpolated to the grid associated with updraft        
        
        tindex.append(w_obj)            
                                        
    if( len(p_vars['w']) < 1 ):

        print("\n ---> Compute_Obj_Profiles found no objects...returning zero...\n")
        return np.zeros((zhgts.shape[0],1,1))

    else:
        
        print("\n Number of selected updraft profiles:  %d \n Number of labeled objects:  %d\n" % (w_obj, all_obj))
             
        for key in p_vars:
            p_vars[key] = np.squeeze(np.asarray(p_vars[key]), axis=1).transpose()
            
        p_vars['tindex'] = tindex
        p_vars['size']   = np.asarray(slist, dtype=np.float32)

        return p_vars 
        
#-------------------------------------------------------------------------------

def getobjdata(path, model_type='wrf', vars=['hgt', 'w', 'pres'], file_pattern=None):
    
    print("processing model run:  %s \n" % run_dir)
    
    if model_type == 'fv3_solo' or model_type =='solo':

        return read_solo_fields(path, file_pattern=file_pattern,
                              vars=vars, ret_dbz=True)
    
    if model_type == 'wrf':
    
        return read_wrf_fields(path, file_pattern=file_pattern,
                              vars=vars, ret_dbz=True)
    
    if model_type == 'cm1':
        
        returnread_cm1_fields(path, file_pattern=file_pattern,
                              vars=vars, ret_dbz=True)
    
    
