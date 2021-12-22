#!/usr/bin/env python2.7

import argparse
import numpy as np
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
matplotlib.rcParams['axes.linewidth'] = 1.5

#--- A Simple Script to Plot All Avaialble Spectra Dependent upon
#--- Model Output and Variable
parser = argparse.ArgumentParser()
parser.add_argument("var",type=str,help = 'The variable you plan to plott')
parser.add_argument("path",type=str,help = 'The location where the spectra are stored')
arguments = parser.parse_args()

fig = plt.figure(figsize=(10,12))

#--- Try All Different Models
path = '%s/spectra1d_hrrr_%s.npz'%(arguments.path,arguments.var)
try:
   data = np.load(path)
   variance = data['field']
   lenscale_bands = data['wavelength']
   plt.loglog((lenscale_bands[1:]/1000.), variance,'k', alpha=1.0, linewidth=(5.0), label='HRRR %s'%arguments.var.upper())
except:
   print('No HRRR Data')        


path = '%s/spectra1d_fv3_%s.npz'%(arguments.path,arguments.var)
try:
   data = np.load(path)
   variance = data['field']
   lenscale_bands = data['wavelength']
   plt.loglog((lenscale_bands[1:]/1000.), variance,'r', alpha=1.0, linewidth=(5.0), label='EMC FV3 %s'%arguments.var.upper()) 
except:
   print('No FV3 Data')

path = '%s/spectra1d_GFDL_%s.npz'%(arguments.path,arguments.var)
try:
   data = np.load(path)
   variance = data['field']
   lenscale_bands = data['wavelength']
   plt.loglog((lenscale_bands/1000.), variance,'g', alpha=1.0, linewidth=(5.0), label='GFDL FV3 %s'%arguments.var.upper())
except:
   print('No GFDL Data')

path = '%s/spectra1d_gsl_%s.npz'%(arguments.path,arguments.var)
try:
   data = np.load(path)
   variance = data['field']
   lenscale_bands = data['wavelength']
   plt.loglog((lenscale_bands/1000.), variance,'b', alpha=1.0, linewidth=(5.0), label='GSL FV3 %s'%arguments.var.upper())
except:
   print('No GSL Data')


path = '%s/spectra1d_NAM_%s.npz'%(arguments.path,arguments.var)
try:
   print(path)
   data = np.load(path)
   variance = data['field']
   lenscale_bands = data['wavelength']
   print('HEERE')
   print('MAx = ',np.amax(variance))
   plt.loglog((lenscale_bands[:]/1000.), variance,'gold', alpha=1.0, linewidth=(5.0), label='NAM %s'%arguments.var.upper())
except:
   print('No NAM Data')


plt.xlim([lenscale_bands[1]/1000.,6.])      
plt.xlim([2000.,6.])
wavenumber = 1/(lenscale_bands)
lindborg_II = (5.0e-4*np.power(wavenumber, -5./3.))
plt.loglog(lenscale_bands[1:]/1000., lindborg_II[1:], color='gray',linestyle='-.',label='k$^{-5/3}$')
plt.loglog([18,18],[1E-10,1E30],color='#00a329',linestyle='--',zorder=0,label='6dx')


if arguments.var in ['u','v']:
   plt.ylim([1E0,1E8])
elif arguments.var in ['TKE']:
   plt.ylim([1E4,1E14])
else:
   plt.ylim([1E0,5E5])

plt.legend(loc=3)
plt.xlabel('Wavelength(km)')
plt.ylabel('Normalized Variance')
plt.savefig('1Dspectra_%s.pdf'%arguments.var,dpi=300)
