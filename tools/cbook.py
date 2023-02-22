import numpy as np
import netCDF4 as ncdf
import matplotlib.pyplot as plt
import xarray as xr
import glob as glob
import os as os
import glob
import sys as sys
import matplotlib as mpl

from datetime import datetime
import cftime
import pickle

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

#--------------------------------------------------------------------------------------------------

def write_Z_profile(data, model='WRF', tindex=0, iloc=0, jloc=0,  data_keys = ['hgt', 'press', 'theta', 'den', 'pert_p']):
    
    try:
        from columnar import columnar
    except:
        print("Need to install columnar into conda, exiting")
        return
    
    print('#-----------------------------#')
    print('          %s                  ' % model)
    print('#-----------------------------#')
    
    newlist = []
    newlist.append(np.arange(data['z3d'].shape[1]).tolist())
    
    headers = ['level']
    
    for key in data_keys:
        
        newlist.append(data[key][tindex,:,iloc,jloc].tolist())
        headers.append(key)

    # need to rearange all the data into row like an excel spreadsheet
    
    row_data = [list(x) for x in zip(*newlist)]
        
    table = columnar(row_data, headers, no_borders=True)
    print(table)

#==========================================================================================================
#
# GENERATE IDEAL PROFILES
#
#==========================================================================================================
def generate_ideal_profiles(run_dir, model_type='wrf', filename=None, w_thresh = 5.0, cref_thresh = 45., 
                            min_pix=1, percentiles=None, compDBZ=False, **kwargs):
    
    print("processing model run:  %s \n" % run_dir)
    
    if model_type == 'solo' or if model_type == 'fv3_solo':

        ds = read_solo_ 
        profiles = compute_obj_profiles(w, dbz, pres, z3d, w_thresh = w_thresh, cref_thresh = cref_thresh, 
                                        min_pix=min_pix, percentiles=percentiles, 
                                        extra_vars={'temp': temp, 'theta':theta, 'pert_t': pert_t, 
                                                    'pert_th':pert_th, 'qv': qv,
                                                    'pert_p': pert_p,  'pert_pss': pert_pss},
                                        **kwargs)
        ds.close()

        return profiles
    
    if model_type == 'wrf':
        
        def open_mfdataset_list(data_dir, pattern, skip=0):
            """
            Use xarray.open_mfdataset to read multiple netcdf files from a list.
            """
            filelist = os.path.join(data_dir,pattern)
            
            if skip > 0:
                filelist = filelist[0:-1:skip]
                
            return xr.open_mfdataset(filelist, combine='nested', concat_dim=['Time'], parallel=True)
    
        if filename == None:
            print("Reading:  %s " % os.path.join(run_dir,"wrfout_d01_0001-01-01_00:00:00"))
            filename = os.path.join(run_dir,"wrfout_d01_0001-01-01_00:00:00")
            ds = xr.open_dataset(filename,decode_times=False)
        else:
            ds   = open_mfdataset_list(run_dir,  "wrfout*")
              
        if not os.path.exists(os.path.join(run_dir, 'wrf_dbz.npy')):
                
            print('\n WRF output is missing REFL_10CM --> computing')

            pbase  = ds.PB.values
            pres   = ds.P.values + ds.PB.values

            theta  = ds.T.values + 300.
            pii    = (pres/100000.)**0.286
            temp   = theta*pii
            den    = pres/(temp*_Rgas)

            qv     = ds.QVAPOR.values
            qc     = ds.QCLOUD.values
            qr     = ds.QRAIN.values
            qi     = np.zeros_like(qv)
            qs     = np.zeros_like(qv)
            qg     = np.zeros_like(qv)
            dbz    = np.zeros_like(qv)

            for n in np.arange(dbz.shape[0]):

                dbz[n] = cmpref.calcrefl10cm(qv[n], qc[n], qr[n], qs[n], qg[n], temp[n], pres[n])
                dbznew  = dbz_from_q(den[n], qr[n])

                print(n, dbz[n].max(), dbz[n].min())
                print(n, dbznew.max(), dbznew.min())

#             ds_new = xr.DataArray( data=dbz, name = 'REFL_10CM', \
#                                    dims   = ['Time','bottom_top','south_north','west_east'], \
#                                    coords = dict(Time=(['Time'], ds['Times'].values),
#                                                  bottom_top  = (['ZNU'], ds['ZNU'].values),
#                                                  south_north = (['XLAT'], ds['XLAT'].values),
#                                                  west_east   = (['XLONG'], ds['XLONG'].values) ) )

#             ds_new.to_netcdf(os.path.join(run_dir, 'wrf_dbz.nc'), mode='w')

            with open(os.path.join(run_dir, 'wrf_dbz.npy'), 'wb') as f:
                np.save(f, dbz)
            
            print("\nWrote file out: %s" % os.path.join(run_dir, 'wrf_dbz.npy'))
            
            # plt.imshow(dbz[10,10])
            # plt.show()

        else:
            
            print("\nReading external DBZ file: %s" % os.path.join(run_dir, 'wrf_dbz.npy'))

            with open(os.path.join(run_dir, 'wrf_dbz.npy'), 'rb') as f:
                dbz = np.load(f)
                
            print("\nShape of DBZ array: " ,dbz.shape)
            
            # plt.imshow(dbz[10,10])
            # plt.show()
                
            # for n in np.arange(dbz.shape[0]):
            #     print(n, dbz.max(), dbz.min())
                
        w      = ds.W.values
        w      = 0.5*(w[:,1:,:,:] + w[:,:-1,:,:])
        pbase  = ds.PB.values
        pert_p = ds.P.values
        pres   = pbase + pert_p

        theta  = ds.T.values + 300.
        pii    = (pres/100000.)**0.286
        temp   = theta*pii
        qv     = ds.QVAPOR.values
        
        temp1d  = temp[0,:,-1,-1]
        pert_t  = temp - np.broadcast_to(temp1d[np.newaxis, :, np.newaxis, np.newaxis], w.shape)
        theta1d = theta[0,:,-1,-1]
        pert_th = theta - np.broadcast_to(theta1d[np.newaxis, :, np.newaxis, np.newaxis], w.shape)

        z    = ds.PHB.values/9.806
        z3d  = 0.5*(z[:,1:,:,:] + z[:,:-1,:,:])
        
        profiles = compute_obj_profiles(w, dbz, pres, z3d, w_thresh = w_thresh, cref_thresh = cref_thresh, 
                                        min_pix=min_pix, percentiles=percentiles, 
                                        extra_vars={'temp': temp, 'theta':theta, 'pert_t': pert_t, 
                                                    'pert_th':pert_th, 'qv': qv,
                                                    'pert_p': pert_p},
                                        **kwargs)
        ds.close()

        return profiles
    
    if model_type == 'fv3':
        
        def open_mfdataset_list(data_dir, pattern):
            """
            Use xarray.open_mfdataset to read multiple netcdf files from a list.
            """
            filelist = os.path.join(data_dir,pattern)
            return xr.open_mfdataset(filelist, combine='nested', concat_dim=['time'], parallel=True)
    
        ds   = open_mfdataset_list(run_dir,   "*.nc")
        
        w    = ds.W.values
        w    = 0.5*(w[:,1:,:,:] + w[:,:-1,:,:])
        dbz  = ds.REFL_10CM.values
        pres = ds.P.values
        z    = ds.PHB.values/9.806
        z3d  = 0.5*(z[:,1:,:,:] + z[:,:-1,:,:])

        profiles = compute_obj_profiles(w, dbz, pres, z3d, w_thresh = w_thresh, cref_thresh = cref_thresh, 
                                        min_pix=min_pix, percentiles=percentiles, 
                                        extra_vars={'temp': temp, 'theta':theta}, **kwargs)
        ds.close()

        return profiles
    
    if model_type == 'cm1':
        
        def open_mfdataset_list(data_dir, pattern):
            """
            Use xarray.open_mfdataset to read multiple netcdf files from a list.
            """
            filelist = os.path.join(data_dir,pattern)
            return xr.open_mfdataset(filelist, parallel=True)
    
        if filename == None:
            print("Reading:  %s " % os.path.join(run_dir,"cm1out.nc"))
            ds = xr.open_dataset(os.path.join(run_dir,"cm1out.nc"),decode_times=False)
        else:
            ds = open_mfdataset_list(run_dir,  "cm1out_*.nc")
        
        w      = ds.winterp.values
        z      = ds.zh.values * 1000. # cm1 heights are stored in km
        z3d    = np.broadcast_to(z[np.newaxis, :, np.newaxis, np.newaxis], w.shape)
        pres   = ds.prs.values
        theta  = ds.th.values
        pii    = (pres/100000.)**0.286
        temp   = theta*pii
        pert_p = ds.prspert.values
        
        temp1d  = temp[0,:,-1,-1]
        pert_t  = temp - np.broadcast_to(temp1d[np.newaxis, :, np.newaxis, np.newaxis], w.shape)
        theta1d = theta[0,:,-1,-1]
        pert_th = theta - np.broadcast_to(theta1d[np.newaxis, :, np.newaxis, np.newaxis], w.shape)
        qv      = ds.qv.values

        if os.path.exists(os.path.join(run_dir, 'cm1_dbz.nc')):

            print("\nReading existing DBZ file in directory")

            ds2 = xr.load_dataset(os.path.join(run_dir, 'cm1_dbz.nc'))
            dbz = ds2['REFL_10CM'].values

        else:

            print("\n  ...No existing DBZ file found, computing a new 4D DBZ\n")

            den   = pres/(temp*_Rgas)
            qc    = ds.qc.values
            qr    = ds.qr.values
            qi    = np.zeros_like(qv)
            qs    = np.zeros_like(qv)
            qg    = np.zeros_like(qv)
            dbz   = np.zeros_like(qv)

            for n in np.arange(dbz.shape[0]):

                dbz[n]  = cmpref.calcrefl10cm(qv[n], qc[n], qr[n], qs[n], qg[n], temp[n], pres[n])
                #dbznew  = dbz_from_q(den[n], qr[n])

                # print(n, dbz[n].max(), dbz[n].min())
                # print(n, dbznew.max(), dbznew.min())

            ds_new = xr.DataArray( data=dbz, name = 'REFL_10CM', \
                                   dims   = ['time','zh','yh','xh'], \
                                   coords = dict(time=(['time'], ds['time'].values),
                                                 zh = (['zh'], ds['zh'].values),
                                                 yh = (["yh"], ds['yh'].values),
                                                 xh = (["xh"], ds['xh'].values) ) )

            ds_new.to_netcdf(os.path.join(run_dir, 'cm1_dbz.nc'), mode='w')

        profiles = compute_obj_profiles(w, dbz, pres, z3d, w_thresh = w_thresh, cref_thresh = cref_thresh, 
                                        min_pix=min_pix, percentiles=percentiles, 
                                        extra_vars={'temp': temp, 'theta':theta, 'pert_t': pert_t, 
                                                    'pert_th':pert_th, 'qv': qv,
                                                    'pert_p': pert_p},
                                        **kwargs)
        
        ds.close()

        return profiles

#-------------------------------------------------------------------------------
def compute_obj_profiles(W, DBZ, PRES, Z, w_thresh = 3.0, cref_thresh = 45., min_pix=5, percentiles=None, 
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
   
    # thanks to Larissa Reames for suggestions on how to do this (and figuring out what to do!)
    
    mask_cref   = np.where(DBZ.max(axis=1) > cref_thresh, True, False)
    mask_w_3d   = np.where(PRES < 70000.0, W, np.nan)
    
    mask_w_2d   = np.nanmax(mask_w_3d, axis=1)
    mask_w_cref = (mask_w_2d > w_thresh) & mask_cref
    f_mask      = mask_w_cref.astype(np.int8)
        
    tindex   = [0]
    
    wlist    = []
    slist    = []
    
    vars = {}
    vars['dbz']  = DBZ
    vars['pres'] = PRES
    vars['w']    = W
    
    p_vars = {}
    p_vars['dbz']  = []
    p_vars['pres'] = []
    p_vars['w']    = []
    
    if extra_vars != None:
        for key in extra_vars:
            vars[key]   = extra_vars[key]  # this should be an array same shape as W, PRES, DBZ
            p_vars[key] = []
    
    all_obj  = 0
    w_obj    = 0
    
    for n in np.arange(W.shape[0]): # loop over number of time steps.            
        
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
                           
                            zraw    = Z[n,:,jloc,iloc]             # get z_raw profiles
                            
                            for key in p_vars:                     # loop through dictionary of variables.
                                tmp = vars[key][n,:,jloc,iloc]
                                profile = interp3dz_np(tmp.transpose(), zraw.transpose(), zhgts, nthreads = _nthreads)
                                p_vars[key].append([profile.mean(axis=(1,))],)

                                if key == 'w':
                                    slist.append(tmp.shape[0])
                            
                            # returns a columns of variables interpolated to the grid associated with updraft
                        
                                
        
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

def getobjdata(run_dir, model_type='wrf', filename=None):
    
    print("processing model run:  %s \n" % run_dir)
    
    if model_type == 'fv3_solo':

        if filename != None:
            print("Reading:  %s " % os.path.join(run_dir,filename))
            ds = xr.open_dataset(os.path.join(run_dir,filename),decode_times=False)
        else:
            ds = xr.open_dataset(os.path.join(run_dir, "atmos_hifreq.nc"), decode_times=False)

        z    = ds.delz.values[:,::-1,:,:]
        z    = np.cumsum(z,axis=1)

        w    = ds.dzdt.values[:,::-1,:,:]
        pres = ds.nhpres.values[:,::-1,:,:]
        
        dbz  = ds.refl_10cm.values[:,::-1,:,:]
        
        ds.close()

        return {'z': z, 'w': w, 'pres': pres, 'dbz':dbz}
    
    if model_type == 'wrf':
        
        def open_mfdataset_list(data_dir, pattern, skip=0):
            """
            Use xarray.open_mfdataset to read multiple netcdf files from a list.
            """
            filelist = os.path.join(data_dir,pattern)
            
            if skip > 0:
                filelist = filelist[0:-1:skip]
                
            return xr.open_mfdataset(filelist, combine='nested', concat_dim=['Time'], parallel=True)
    
        if filename == None:
            print("Reading:  %s " % os.path.join(run_dir,"wrfout_d01_0001-01-01_00:00:00"))
            filename = os.path.join(run_dir,"wrfout_d01_0001-01-01_00:00:00")
            ds = xr.open_dataset(filename,decode_times=False)
        else:
            ds   = open_mfdataset_list(run_dir,  "wrfout*")
              
        if not os.path.exists(os.path.join(run_dir, 'wrf_dbz.npy')):
                
            print('\n WRF output is missing REFL_10CM --> computing')

            pbase  = ds.PB.values
            pres   = ds.P.values + ds.PB.values

            theta  = ds.T.values + 300.
            pii    = (pres/100000.)**0.286
            temp   = theta*pii

            qv     = ds.QVAPOR.values
            qc     = ds.QCLOUD.values
            qr     = ds.QRAIN.values
            qi     = np.zeros_like(qv)
            qs     = np.zeros_like(qv)
            qg     = np.zeros_like(qv)
            dbz    = np.zeros_like(qv)

            for n in np.arange(dbz.shape[0]):

                dbz[n] = cmpref.calcrefl10cm(qv[n], qc[n], qr[n], qs[n], qg[n], temp[n], pres[n])
                print(n, dbz.max(), dbz.min())

#             ds_new = xr.DataArray( data=dbz, name = 'REFL_10CM', \
#                                    dims   = ['Time','bottom_top','south_north','west_east'], \
#                                    coords = dict(Time=(['Time'], ds['Times'].values),
#                                                  bottom_top  = (['ZNU'], ds['ZNU'].values),
#                                                  south_north = (['XLAT'], ds['XLAT'].values),
#                                                  west_east   = (['XLONG'], ds['XLONG'].values) ) )

#             ds_new.to_netcdf(os.path.join(run_dir, 'wrf_dbz.nc'), mode='w')

            with open(os.path.join(run_dir, 'wrf_dbz.npy'), 'wb') as f:
                np.save(f, dbz)
            
            print("\nWrote file out: %s" % os.path.join(run_dir, 'wrf_dbz.npy'))
            
            # plt.imshow(dbz[10,10])
            # plt.show()

        else:
            
            print("\nReading external DBZ file: %s" % os.path.join(run_dir, 'wrf_dbz.npy'))

            with open(os.path.join(run_dir, 'wrf_dbz.npy'), 'rb') as f:
                dbz = np.load(f)
                
            print("\nShape of DBZ array: " ,dbz.shape)
            
            # plt.imshow(dbz[10,10])
            # plt.show()
                
            # for n in np.arange(dbz.shape[0]):
            #     print(n, dbz.max(), dbz.min())
                
        w      = ds.W.values
        w      = 0.5*(w[:,1:,:,:] + w[:,:-1,:,:])
        pbase  = ds.PB.values
        pres   = ds.P.values + ds.PB.values

        theta  = ds.T.values + 300.
        pii    = (pres/100000.)**0.286
        temp   = theta*pii

        z    = ds.PHB.values/9.806
        z    = 0.5*(z[:,1:,:,:] + z[:,:-1,:,:])
                
        ds.close()

        return {'z': z, 'w': w, 'pres': pres, 'dbz':dbz}
    
    if model_type == 'fv3':
        
        def open_mfdataset_list(data_dir, pattern):
            """
            Use xarray.open_mfdataset to read multiple netcdf files from a list.
            """
            filelist = os.path.join(data_dir,pattern)
            return xr.open_mfdataset(filelist, combine='nested', concat_dim=['time'], parallel=True)
    
        ds   = open_mfdataset_list(run_dir,   "*.nc")
        
        w    = ds.W.values
        w    = 0.5*(w[:,1:,:,:] + w[:,:-1,:,:])
        dbz  = ds.REFL_10CM.values
        pres = ds.P.values
        z    = ds.PHB.values/9.806
        z    = 0.5*(z[:,1:,:,:] + z[:,:-1,:,:])

        profiles = compute_obj_profiles(w, dbz, pres, z, w_thresh = w_thresh, cref_thresh = cref_thresh, 
                                        min_pix=min_pix, percentiles=percentiles)
        
        ds.close()

        return profiles
    
    if model_type == 'cm1':
        
        def open_mfdataset_list(data_dir, pattern):
            """
            Use xarray.open_mfdataset to read multiple netcdf files from a list.
            """
            filelist = os.path.join(data_dir,pattern)
            return xr.open_mfdataset(filelist, parallel=True)
    
        if filename == None:
            print("Reading:  %s " % os.path.join(run_dir,"cm1out.nc"))
            ds = xr.open_dataset(os.path.join(run_dir,"cm1out.nc"),decode_times=False)
        else:
            ds = open_mfdataset_list(run_dir,  "cm1out_*.nc")
        
        w    = ds.winterp.values
        z    = ds.zh.values * 1000. # heights are in km
        z3d  = np.broadcast_to(z[np.newaxis, :, np.newaxis, np.newaxis], w.shape)
        pres = ds.prs.values

        if os.path.exists(os.path.join(run_dir, 'cm1_dbz.nc')):

            print("\nReading existing DBZ file in directory")

            ds2 = xr.load_dataset(os.path.join(run_dir, 'cm1_dbz.nc'))
            dbz = ds2['REFL_10CM'].values

        else:

            print("\n  ...No existing DBZ file found, computing a new 4D DBZ\n")

            theta = ds.th.values
            pii   = (pres/100000.)**0.286
            temp  = theta*pii
            qv    = ds.qv.values
            qc    = ds.qc.values
            qr    = ds.qr.values
            qi    = np.zeros_like(qv)
            qs    = np.zeros_like(qv)
            qg    = np.zeros_like(qv)
            dbz   = np.zeros_like(qv)

            for n in np.arange(dbz.shape[0]):

                dbz[n]  = cmpref.calcrefl10cm(qv[n], qc[n], qr[n], qs[n], qg[n], temp[n], pres[n])
                print(n, dbz[n].max(), dbz[n].min())

            ds_new = xr.DataArray( data=dbz, name = 'REFL_10CM', \
                                   dims   = ['time','zh','yh','xh'], \
                                   coords = dict(time=(['time'], ds['time'].values),
                                                 zh = (['zh'], ds['zh'].values),
                                                 yh = (["yh"], ds['yh'].values),
                                                 xh = (["xh"], ds['xh'].values) ) )

            ds_new.to_netcdf(os.path.join(run_dir, 'cm1_dbz.nc'), mode='w')
        
        ds.close()

        return {'z': z, 'w': w, 'pres': pres, 'dbz':dbz}
    
    
    
    
