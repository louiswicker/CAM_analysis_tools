#!/usr/bin/env python
import glob
import os
import numpy as np
from netCDF4 import Dataset
import sharppy.sharptab.profile as profile
import sharppy.sharptab.params as params
import datetime
from test_wof_sounding_plot import plot_wof
import sharppy.sharptab.utils as utils
import sharppy.sharptab.thermo as thermo
import argparse
import time
import multiprocessing
import sys
import warnings
from analsnd import analsnd_mod
from idealized_sounding_fcts import mccaul_weisman

warnings.filterwarnings("ignore",message="converting a masked element to nan",append=True)
warnings.filterwarnings("ignore",message="overflow encountered in multiply",append=True)
warnings.simplefilter(action="ignore",category=RuntimeWarning)
warnings.simplefilter(action="ignore",category=UserWarning)


grav = 9.806
Cp   = 1004.
Rgas = 287.04
Rwat = 461.5
Lv = 2.501E6

def sounding(gz, dz, theta_sfc = 300., qv_sfc = 14.):
    
    #define parameters
    z_trop     = 12000.
    theta_trop = 343.
    temp_trop  = 213.

    
    # compute theta profile
    theta = 0.0*gz
    t1    = theta_sfc + (theta_trop - theta_sfc)*((gz/z_trop)**1.25)
    t2    = theta_trop * np.exp(grav * (gz - z_trop) / (Cp * temp_trop))   
    theta = np.where(gz <= z_trop, t1, t2)
    
    # compute rh profile
    rh    = np.where(gz <= z_trop, 1.0 - 0.75*((gz/z_trop)**1.25), 0.25)
    
    
    
    # integrate hydrostatic eq
    pi    = 0.0*gz
    p     = 0.0*gz
    pi[0] = 1.0 - 0.5*dz[0]*grav/(Cp*theta_sfc)
    
    for k in np.arange(1,gz.shape[0]):
        pi[k] = pi[k-1] - 2.0*dz[k]*grav/(Cp*(theta[k-1]+theta[k]))
    
    # compute pressure (Pa) profile
    p = 1.0e5 * pi**(3.5088)
    t = pi*theta - 273.16
    
    # compute qv profile
    qvs = (380./p) * np.exp(17.27*((pi*theta - 273.16)/(pi*theta-36.)))
    qv  = qvs*rh
    qv  = np.where( qv > qv_sfc/1000., qv_sfc/1000., qv)
    
    # recompute RH so that it reflects the new qv_sfc limit - needed to dewpoint computation.
    rh  = qv/qvs
    
    ess = 6.112 * np.exp(17.67 * t / (t + 243.5))
    val = np.log(rh * ess/6.112)
    td  = 243.5 * val / (17.67 - val)
    
    den = p / (t+273.16 * Rgas)
    
    return theta, t, qv, p, td

def shear(gz, type=1, shear=12.5, depth=2500.):

    if type == 0.0:
        return np.zeros_like(gz), np.zeros_like(gz)
    
    if type == 1:  # 1D WK squall line or supercell shear...
        
        scale = gz / depth
        
        u = np.where(scale <= 1.0, shear*scale, shear)
        
        return u, np.zeros_like(gz)
    
def zgrid(nz, height = 20000., stag=False): 
    dz = height / float(nz)
    if stag:
        dz = height / float(nz-1)
        return np.arange(nz)*dz
    else:
        return (np.arange(nz)+0.5)*dz
        
def write_2_file(p0, th0, q0, z, theta, qv, u, v, filename='wk.sounding'):
    
    with open(filename, 'w') as f:
        f.write("%12.4f %12.4f  %12.4f \n" % (p0, th0, q0))
        for k in np.arange(z.shape[0]):
            f.write("%12.4f  %12.4f  %12.4f  %12.4f  %12.4f\n" % (z[k], theta[k], 1000.*qv[k], u[k], v[k]))
        
    f.close()
        
nz = 300
ztop = 30000.
gze = zgrid(nz+1, height = ztop, stag=True)
hgt = zgrid(nz,  height = ztop)
dz  = gze[1:] - gze[:-1]

zcape = 12.0E3
ztrop = 12.0E3

CAPEs =(1000, 1500, 2000, 2500, 3000, 3500)
#CAPE = 2000.
qvs = (11., 12., 13., 14., 15., 16.)
m = 1.8
us1 = 12.5
pblthe = 343.0
#def mccaul_weisman(z, E=2000.0, m=2.2, H=12500.0, z_trop=12000.0, RH_min=0.1, p_sfc=1e5,
#                   T_sfc=300.0, thetae_pbl=335.0, pbl_lapse=0.009, crit_lapse=0.0095, 
#                   pbl_depth=None, lr=0.0001):

for i, CAPE in enumerate(CAPEs):
  fname_plot = 'poly_sounding_C%d'%(CAPE)
  fname_out = 'poly_input_sounding_C%d'%(CAPE)
  profile_mw = mccaul_weisman(hgt,CAPE,m,zcape,ztrop,thetae_pbl=pblthe,RH_min=0.25,pbl_lapse = 0.008)
#Returns     
#        thermo_prof : pd.DataFrame
#        MW01 sounding
#            z : height (m)
#            prs : pressure (Pa)
#            qv : water vapor mixing ratio (kg / kg)
#            T : temperature (K) 
#            Td : dewpoint (K)
  z = profile_mw['z']
  p = profile_mw['prs']
  t = profile_mw['T'] 
  th = thermo.theta(p/100.,t-273.15)+273.15
  qv = profile_mw['qv']
  tdc = profile_mw['Td'] - 273.15
  tc = t - 273.15

#fname_plot = 'sounding_qv%d'%(q)
#fname_out = 'input_sounding_q%d'%(q)
#th,tc,qv,p,tdc = sounding(hgt,dz,qv_sfc=q)
#tdc = td #thermo.temp_at_mixrat(qv*1000., p/100.)
#t = thermo.theta(1000.,th, p/100.)
  u,v = shear(hgt)

  ukt = u*1.94384
  vkt = v*1.94384
  omega = np.full(v.shape,0.0)

  start_date = '201905202100'
  idateobj = datetime.datetime.strptime(start_date,'%Y%m%d%H%M')
  vdateobj = idateobj 
  member_profs = np.empty(1, dtype=object)
  prof = profile.create_profile(profile='convective', pres=p/100., hght=hgt, tmpc=tc, \
					dwpc=tdc, u=u, v=v, omega=omega, missing=-9999, latitude=35.0, strictQC=False,date=vdateobj )
  prof.pw = params.precip_water(prof)
  member_profs[0] = prof
  members = {'hght': hgt, 'pres': p, 'tmpc': tc, 'dwpc': tdc, 'u': ukt, 'v': vkt, 'member_profs': member_profs}

  plot_wof(prof, members, fname_plot , 0.0, 0.0, idateobj, vdateobj,x_pts=(0,0),y_pts=(0,0))
  write_2_file(p[0]/100.,th[0],qv[0]*1000.,hgt,th,qv,u,v,filename=fname_out)
    
    
