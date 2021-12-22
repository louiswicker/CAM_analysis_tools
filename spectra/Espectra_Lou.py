#!/usr/bin/env python
#################################################
# File: energy.py                               #
# Date: 26 Oct 2011                             #
# Auth: Jeremy A. Gibbs                         #                                  
# Desc: Calculates 2D energy spectra            #
#################################################
import os, sys, argparse, cmath, spectra
import pylab as pl
from mpi4py import MPI
import numpy as np
import netCDF4 as nc4
from scipy import integrate
from matplotlib import colors, ticker
import scipy.ndimage as ndimage

##################
# Energy Spectra #
##################
parser = argparse.ArgumentParser()
parser.add_argument("model",type=str,help='The Model Output You Want to Work With (fv3, hrrr, GFDL, gsl, NAM)')
parser.add_argument("var",type=str,help='Evaluated Variable (u,v,w,TKE)')
arguments = parser.parse_args()

imag = cmath.sqrt(-1)
pi   = np.pi

#################
# Open the file #
#################
print('--- opening MicroHH output file')
#--- Loop Over These Times and Average
times = [12,13,14,15,16,17,18,19,20,21,22,23,24]
ntimes = len(times)
z = 1
model = arguments.model #'GFDL' #'hrrr'
#--- JDL Loop over the three times and average spectra
for tindex,time in enumerate(times):
   #--- Pick From The Available Model Output Select Variables (u,v,w,TKE)
   if model == 'fv3':
      dumpfile = nc4.Dataset('/scratch/wicker/SFE/fv3_emc_nc/fv3lam_2021050400f%03d.nc'%time)
      lev = 2
      if arguments.var == 'w':
         varname = 'MAXUVV_P8_2L100_GLC0_max1h'
         data = dumpfile.variables[varname][:,:].astype("float64")
         data = data[None,None,:,:]
      elif arguments.var in  ['u','v']:
         if arguments.var == 'u':
            varname = 'UGRD_P0_L100_GLC0'
         elif arguments.var == 'v':
            varname = 'VGRD_P0_L100_GLC0'
         data = dumpfile.variables[varname][:,:].astype("float64")
         data = data[None,None,:,:]
      elif arguments.var in ['TKE']:
         u = dumpfile.variables['UGRD_P0_L100_GLC0'][:,:].astype("float64")
         v = dumpfile.variables['VGRD_P0_L100_GLC0'][:,:].astype("float64")
         w = dumpfile.variables['MAXUVV_P8_2L100_GLC0_max1h'][:,:].astype("float64")
         data = (u**2.) + (v**2.) + (w**2.)
         data = data[None,None,:,:]

   elif model == 'hrrr':
      dumpfile  = nc4.Dataset('/scratch/wicker/SFE/hrrrV4_nc/hrrr_ncep_2021050400f%03d.nc'%time)
      lev = 2
      if arguments.var == 'w':
         varname = 'MAXUVV_P8_2L108_GLC0_max1h'
         data = dumpfile.variables[varname][:,:].astype("float64")
         data = data[None,None,:,:]
      elif arguments.var in ['u','v']:
         if arguments.var == 'u':
            varname = 'UGRD_P0_L100_GLC0'
         elif arguments.var == 'v':
            varname = 'VGRD_P0_L100_GLC0'
         data = dumpfile.variables[varname][lev:lev+1,:,:].astype("float64")
         data = data[None,:,:,:]
      elif arguments.var in ['TKE']:
         u = dumpfile.variables['UGRD_P0_L100_GLC0'][lev,:,:].astype("float64")
         v = dumpfile.variables['VGRD_P0_L100_GLC0'][lev,:,:].astype("float64")
         w = dumpfile.variables['MAXUVV_P8_2L108_GLC0_max1h'][:,:].astype("float64")
         data = (u**2.) + (v**2.) + (w**2.)
         data = data[None,None,:,:]

   elif model == 'GFDL':
      dumpfile  = nc4.Dataset('/scratch/wicker/SFE/GFDL_nc/C-SHiELD_C768n5r10_hwt_CLUEgrid_2021050400f%03d.nc'%time)
      if arguments.var == 'w' :
         varname = 'MAXUVV_P8_2L108_GLC0_max1h'
         data = dumpfile.variables[varname][:,:].astype("float64")
         data = data[None,None,:,:]
      elif arguments.var in ['u','v','TKE']:
         print('Variables not supported in output')

   elif model == 'gsl':
      dumpfile  = nc4.Dataset('/scratch/wicker/SFE/fv3_gsl_nc/RRFS_CONUS.t00z.bgdawpf%03d.tm00.nc'%time)
      if arguments.var == 'w':
         varname = 'MAXUVV_P8_2L100_GLC0_max1h'
         data = dumpfile.variables[varname][:,:].astype("float64")
         data = data[None,None,:,:]
      elif arguments.var in ['u','v','TKE']:
         print('Variables not supported in output')

   elif model == 'NAM':
      dumpfile = nc4.Dataset('/scratch/wicker/SFE/nam_nest_nc/nam_conusnest_2021050400f%03d.nc'%time)
      lev = 1 #--- The height of 500 hPA
      if arguments.var == 'w':
         varname = 'MAXUVV_P8_2L100_GLC0_max1h'
         data = dumpfile.variables[varname][0:1050,0:1750].astype("float64")
         data = data[None,None,:,:]
      elif arguments.var in ['u','v']:
         if arguments.var == 'u':
            varname = 'UGRD_P0_L100_GLC0'
         elif arguments.var == 'v':
            varname = 'VGRD_P0_L100_GLC0'
         data = dumpfile.variables[varname][lev:lev+1,0:1050,0:1750].astype("float64")
         data = data[None,:,:,:]
      elif arguments.var in ['TKE']:
         u = dumpfile.variables['UGRD_P0_L100_GLC0'][lev,0:1050,0:1750].astype("float64")
         v = dumpfile.variables['VGRD_P0_L100_GLC0'][lev,0:1050,0:1750].astype("float64")
         w = dumpfile.variables['MAXUVV_P8_2L100_GLC0_max1h'][0:1050,0:1750].astype("float64")
         data = (u**2.) + (v**2.) + (w**2.)
         data = data[None,None,:,:]
   print('--- obtaining grid information')
   #data = data[np.newaxis]

   #--- JDL Come Back
   dx = 3000.
   dy = 3000.
   [jMax,iMax] = data.shape[2:4]
 

   #jMax = len(les.variables['yh'][ymin:ymax])
   #iMax = len(les.variables['xh'][xmin:xmax])



   #if arguments.rst:
   xdis = np.arange(0,iMax)*dx #les.variables['xh'][xmin:xmax]
   ydis = np.arange(0,jMax)*dy #les.variables['yh'][ymin:ymax]
   mMax = int(iMax / 2)
   nMax = int(jMax / 2)
   iter = 1

   if tindex == 0:
      PLES2 = np.zeros((ntimes,jMax,iMax))
      PLES = np.zeros((ntimes,int(iMax/float(2))+1))
      freq = np.arange(0,iMax/2+1)/float(iMax)
      wavelength = dx/freq
 
   print('--- reading in %s-variable data'%arguments.var)

   #########################
   # Subtract means        #
   # leaving perturbations #
   #########################
   print('--- subtracting 1D mean')
   pert1D = data
   print('pert1D shape = ',pert1D.shape)
   print('JDL pert1D = ',np.amax(pert1D))
   for j in range(0,jMax):
      pert1D[0,0,j,:] = data[0,0,j,:] - np.mean(data[0,0,j,:])
   print("JDL pert1DB = ",pert1D)
   ################
   # Wave numbers #
   ################
   print('--- constructing 1D wavenumbers')
   wn = np.abs(2 * pi * np.fft.fftfreq(iMax,dx))[0:mMax+1]
   dk = wn[1]- wn[0]

   ###########################
   # One-Dimensional Spectra #
   ###########################
   print('--- calculating 1D 1-sided auto-spectral density')
   #PLES_tmp   = spectra.spectra1D_meanrecord(pert1D,dx)
   print('JDL pert1DC = ',pert1D.shape)
   PLES_tmp = spectra.spectra1D_meanplane(pert1D,dx)
   PLES[tindex]  = np.squeeze(PLES_tmp)
   
   #print('JDL PLES[tindex] = ',PLES[tindex])
   print('JDL pert1D = ',pert1D)
   #print('JDL dx =',dx)
   #########################


   # Subtract means        #
   # leaving perturbations #
   #########################
   print('--- subtracting 2D mean')
   pert2D = data
   pert2D[0,0,:,:] = data[0,0,:,:] #- np.mean(data[0,0,:,:])

   ################
   # Wave numbers #
   ################
   print('--- constructing 2D wavenumbers')
   wnx    = np.fft.fftshift(2 * pi * np.fft.fftfreq(iMax,dx))


   # --- JDL Additions
   #print('JDL PLES shape = ',PLES.shape)
   #print('JDL wnx shape = ',wnx.shape)
   #print('iMax = ',iMax)
   #print(wnx[100:])
   #print('0 loc = ',PLES[1])
   #freq = np.arange(0,iMax/2+1)/float(iMax)
   #wavelength_Jon = 250./freq
   #pl.plot((wavelength_Jon)/1000.,PLES)
   #pl.xlim([30,0.5])
   #pl.loglog()
   #pl.show()

   #--- JDL END ADDITIONS
   #print('iMax =',iMax)
   #for windex,w in enumerate(np.fft.fftfreq(iMax,dx)):
   #   print('%s index = %s'%(str(windex),str(w)))
   wny    = np.fft.fftshift(2 * pi * np.fft.fftfreq(jMax,dx))
   dkx    = wnx[1]-wnx[0]
   dky    = wny[1]-wny[0]

   ###########################
   # Two-Dimensional Spectra #
   ###########################
   print('--- calculating 2D auto-spectral density')
   #print('JDL pert2D shape = ',pert2D.shape)
   print('JDL MAX PERT =',np.amax(pert2D))
   PLES2_tmp = spectra.spectra2D(pert2D,dx)
   PLES2[tindex] = np.squeeze(PLES2_tmp)
   #print('JDL PLES2 shape = ',PLES2.shape)

   ###################################################
   # this moves zero-frequency entry to correct spot #
   ###################################################
   PLES2[tindex] = np.roll(PLES2[tindex],2,axis=1)
   PLES2[tindex] = np.fft.fftshift(PLES2[tindex])

   #############################################
   # Check that spectra integrates to variance #
   #############################################
   # Built-in variance #
   #####################
   variance = np.zeros(jMax,dtype="float64")
   for j in range(0,jMax):
      variance[j] = np.var(pert1D[0,0,j,:])
   variance = np.mean(variance)

   ########################
   # Built-in Integration #
   ########################
   integration = integrate.trapz(PLES[tindex], dx=dk)

   #################
   # Print results #
   #################
   print("------------------------")
   print("-     One-D Test       -")
   print("------------------------")
   print("Variance"                )
   print("------------------------")
   print(variance                  )
   print("------------------------")
   print("Integration"             )
   print("------------------------")
   print(integration               )
   print("------------------------")
   print("-   End One-D Test     -")
   print("------------------------")


   #########################################
   # Check that psd integrates to variance #
   #########################################
   # Built-in variance #
   #####################
   variance = np.var(pert2D)

   ###################
   # Manual variance #
   ###################
   var2  = 0
   for k in range(0,jMax):
      for l in range(0,iMax):
         var2 += (pert2D[0,0,k,l])**2
   var2 = var2 / (iMax*jMax)

   ########################
   # Built-in Integration #
   ########################
   var3 = integrate.trapz( integrate.trapz(PLES2[tindex], dx=dky, axis=1), dx=dkx, axis=0)

   ######################
   # Manual integration #
   ######################
   var4 = 0
   for k in range(1,jMax):
       dky    = wny[k] - wny[k-1]
       for l in range(1,iMax):
           dkx    = wnx[l] - wnx[l-1]
           var4 += PLES2[tindex,k,l] * dkx * dky

   print("------------------------")
   print("-     Two-D Test       -")
   print("------------------------")
   print("Built-in variance"       )
   print("------------------------")
   print(variance                  )
   print("------------------------")
   print("Manual variance"         )
   print("------------------------")
   print(var2                      )
   print("------------------------")
   print("Built-in integration"    )
   print("------------------------")
   print(var3                      )
   print("------------------------")
   print("Manual integration"      )
   print("------------------------")
   print(var4                      )
   print("------------------------")
   print("-   End Two-D Test     -")
   print("------------------------")


#############
# Plotting  #
#############

# filter spectra for smoother contours
PLES2 = np.mean(PLES2,axis=0)
PLES2 = ndimage.gaussian_filter(PLES2, sigma=1.0, order=0)

cMin    = 1E1
cMax    = 1E9
lev_exp = np.arange(np.floor(np.log10(cMin)),np.ceil(np.log10(cMax))+1)
clevs   = np.power(10, lev_exp)
print('CLEVS = ',clevs)
pl.figure(figsize=(7.5,8.5))
ax = pl.subplot(111)
pl.title('2D Spectra',fontsize=12)

#--- Save the following output to plot later
outpath = 'spectra2d_%s_%s.npz'%(model,arguments.var)
np.savez(outpath, field=PLES2, wnx=wnx, wny=wny, clevs=clevs)

outpath = 'spectra1d_%s_%s.npz'%(model,arguments.var)
np.savez(outpath, field = np.mean(PLES,axis=0),wavelength = wavelength)

