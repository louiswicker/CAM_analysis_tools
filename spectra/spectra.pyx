 #!/usr/bin/env python
############################################
# File: spectra.pyx                        #
# Date: 18 Oct 2010                        #
# Auth: Jeremy A. Gibbs                    #
# Desc: cython code to calculate 1-sided   #
#       autospectral density               #
############################################
import cython
cimport cython
import numpy as np
cimport numpy as np
import math

##############################################
# Define complex exponent and sqrt functions #
##############################################
cdef extern from "complex.h":
    double complex cexp(double complex)
    double complex csqrt(double complex)
    float cabs(double complex)
    
################################################
# Define constants and absolute value function #
################################################
cdef extern from "math.h":
    float M_PI
    float abs(float complex)
cdef float pi         = M_PI
cdef double complex i = csqrt(-1)
################################################
# Define array types                           #
################################################
DTYPE32 = np.float32
DTYPE64 = np.float64
DTYPEDC = np.complex128
ctypedef np.float32_t DTYPE32_t
ctypedef np.float64_t DTYPE64_t
ctypedef np.complex128_t DTYPEDC_t

################################################
# Define variance function                     #
################################################
@cython.cdivision(True)
def variance(np.ndarray[DTYPE32_t, ndim=3] varArray):
    
    ################################################
    # Define static types for constants and arrays #
    ################################################
    cdef int kMax = varArray.shape[0]
    cdef int jMax = varArray.shape[1]
    cdef int iMax = varArray.shape[2]
    
    cdef Py_ssize_t t, k, j, i
    
    cdef np.ndarray[DTYPE32_t, ndim=1] planeMean = np.zeros(kMax,dtype=DTYPE32)
    cdef np.ndarray[DTYPE32_t, ndim=1] planeVar  = np.zeros(kMax,dtype=DTYPE32)
    
    #################################################
    # Calculate planar means for each height/time   #
    #################################################
    for k in range(0,kMax):
        for j in range(0,jMax):
            for i in range(0,iMax):
                planeMean[k] = planeMean[k] + ( varArray[k,j,i] / (iMax * jMax ) )
        for j in range(0,jMax):
            for i in range(0,iMax):
                planeVar[k]  = planeVar[k]  + ( ( varArray[k,j,i] - planeMean[k] )**2 / (jMax*iMax) )
    return planeVar
    
################################################
# Define Haar function                     #
################################################
@cython.cdivision(True)
def haar(np.ndarray[DTYPE32_t, ndim=3] varArray, np.ndarray[DTYPE32_t, ndim=1] zArray, float dz):
    
    ################################################
    # Define static types for constants and arrays #
    ################################################
    cdef int kMax = varArray.shape[0]
    cdef int jMax = varArray.shape[1]
    cdef int iMax = varArray.shape[2]
    cdef int bMax = kMax
    cdef float  a = 1000.0
    cdef float b
    cdef float z
    cdef int h
    cdef Py_ssize_t bb, k, j, i
    
    cdef np.ndarray[DTYPE32_t, ndim=3] wlist = np.zeros((bMax,jMax,iMax),dtype=DTYPE32)
    cdef np.ndarray[DTYPE32_t, ndim=3] fh    = np.zeros((kMax,jMax,iMax),dtype=DTYPE32)
    
    for bb in range(bMax):
        
        b = zArray[bb]
    
        for k in range(kMax):
            z = zArray[k]
            
            for j in range(0,jMax):
                for i in range(0,iMax):
                    if (b-a/2. <= z <= b):
                        h = 1
       
                    elif (b <= z <= b+a/2.):
                        h = -1
    
                    else:
                        h = 0

                    fh[k,j,i] = varArray[k,j,i] * h
        
        for j in range(0,jMax):
            for i in range(0,iMax):
                
                # hz = (zArray[-1] - zArray[0]) / kMax
#                 s = 0
#                 s+= fh[0,j,i]/2.0
#                 for z in range(1,kMax):
#                     s += fh[z,j,i]
#                 s += fh[-1,j,i]/2.0
#                 wlist[bb,j,i] = 1/a * s*hz #np.trapz(fh[:,j,i],dx=dz)
#                 #
                wlist[bb,j,i] = 1/a * np.trapz(fh[:,j,i],dx=dz)
    
    return wlist
    
################################################
# Define covariance function                   #
################################################
@cython.cdivision(True)
def covariance(np.ndarray[DTYPE32_t, ndim=3] varArray1, np.ndarray[DTYPE32_t, ndim=3] varArray2):
    
    ################################################
    # Define static types for constants and arrays #
    ################################################
    cdef int kMax = varArray1.shape[0]
    cdef int jMax = varArray1.shape[1]
    cdef int iMax = varArray1.shape[2]
    
    cdef Py_ssize_t t, k, j, i
    
    cdef np.ndarray[DTYPE32_t, ndim=1] planeMean1 = np.zeros(kMax,dtype=DTYPE32)
    cdef np.ndarray[DTYPE32_t, ndim=1] planeMean2 = np.zeros(kMax,dtype=DTYPE32)
    cdef np.ndarray[DTYPE32_t, ndim=1] planeCoVar = np.zeros(kMax,dtype=DTYPE32)
    
    #################################################
    # Calculate planar means for each height/time   #
    #################################################
    for k in range(0,kMax):
        for j in range(0,jMax):
            for i in range(0,iMax):
                planeMean1[k] = planeMean1[k] + varArray1[k,j,i] / (iMax * jMax)
                planeMean2[k] = planeMean2[k] + varArray2[k,j,i] / (iMax * jMax)
        for j in range(0,jMax):
            for i in range(0,iMax):
                planeCoVar[k] = planeCoVar[k] + ( (varArray1[k,j,i] - planeMean1[k]) * (varArray2[k,j,i] - planeMean2[k]) / (jMax*iMax) )
    return planeCoVar
    
###########################################
# Define TKE covariance                   #
###########################################
@cython.cdivision(True)
def tkevariance(np.ndarray[DTYPE32_t, ndim=3] varArray1, np.ndarray[DTYPE32_t, ndim=3] varArray2, np.ndarray[DTYPE32_t, ndim=3] varArray3):
    
    ################################################
    # Define static types for constants and arrays #
    ################################################
    cdef int kMax = varArray1.shape[0]
    cdef int jMax = varArray1.shape[1]
    cdef int iMax = varArray1.shape[2]
    
    cdef Py_ssize_t t, k, j, i
    
    cdef np.ndarray[DTYPE32_t, ndim=1] planeMean1  = np.zeros(kMax,dtype=DTYPE32)
    cdef np.ndarray[DTYPE32_t, ndim=1] planeMean2  = np.zeros(kMax,dtype=DTYPE32)
    cdef np.ndarray[DTYPE32_t, ndim=1] planeMean3  = np.zeros(kMax,dtype=DTYPE32)
    cdef np.ndarray[DTYPE32_t, ndim=1] planeTKEVar = np.zeros(kMax,dtype=DTYPE32)
    
    #################################################
    # Calculate planar means for each height/time   #
    #################################################
    for k in range(0,kMax):
        for j in range(0,jMax):
            for i in range(0,iMax):
                planeMean1[k] = planeMean1[k] + varArray1[k,j,i] / (iMax * jMax)
                planeMean2[k] = planeMean2[k] + varArray2[k,j,i] / (iMax * jMax)
                planeMean3[k] = planeMean3[k] + varArray3[k,j,i] / (iMax * jMax)
        for j in range(0,jMax):
            for i in range(0,iMax):
                planeTKEVar[k] = planeTKEVar[k] + ( (varArray3[k,j,i] - planeMean3[k]) * 0.5*( (varArray1[k,j,i]-planeMean1[k])**2 + (varArray2[k,j,i]-planeMean2[k])**2 + (varArray3[k,j,i]-planeMean3[k])**2) / (jMax*iMax) )
    return planeTKEVar
    
###########################################
# Define TKE covariance                   #
###########################################
@cython.cdivision(True)
def tke(np.ndarray[DTYPE32_t, ndim=3] varArray1, np.ndarray[DTYPE32_t, ndim=3] varArray2, np.ndarray[DTYPE32_t, ndim=3] varArray3):
    
    ################################################
    # Define static types for constants and arrays #
    ################################################
    cdef int kMax = varArray1.shape[0]
    cdef int jMax = varArray1.shape[1]
    cdef int iMax = varArray1.shape[2]
    
    cdef Py_ssize_t t, k, j, i
    
    cdef np.ndarray[DTYPE32_t, ndim=1] planeMean1  = np.zeros(kMax,dtype=DTYPE32)
    cdef np.ndarray[DTYPE32_t, ndim=1] planeMean2  = np.zeros(kMax,dtype=DTYPE32)
    cdef np.ndarray[DTYPE32_t, ndim=1] planeMean3  = np.zeros(kMax,dtype=DTYPE32)
    cdef np.ndarray[DTYPE32_t, ndim=1] planeTKE    = np.zeros(kMax,dtype=DTYPE32)
    
    #################################################
    # Calculate planar means for each height/time   #
    #################################################
    for k in range(0,kMax):
        for j in range(0,jMax):
            for i in range(0,iMax):
                planeMean1[k] = planeMean1[k] + varArray1[k,j,i] / (iMax * jMax)
                planeMean2[k] = planeMean2[k] + varArray2[k,j,i] / (iMax * jMax)
                planeMean3[k] = planeMean3[k] + varArray3[k,j,i] / (iMax * jMax)
        for j in range(0,jMax):
            for i in range(0,iMax):
                planeTKE[k] = planeTKE[k] + ( 0.5*( (varArray1[k,j,i]-planeMean1[k])**2 + (varArray2[k,j,i]-planeMean2[k])**2 + (varArray3[k,j,i]-planeMean3[k])**2) / (jMax*iMax) )
    return planeTKE

###########################################
# Define TKE covariance                   #
###########################################
@cython.cdivision(True)
def stats(np.ndarray[DTYPE32_t, ndim=3] varArray1):
    
    ################################################
    # Define static types for constants and arrays #
    ################################################
    cdef int tMax = varArray1.shape[0]
    cdef int jMax = varArray1.shape[1]
    cdef int iMax = varArray1.shape[2]
    
    cdef Py_ssize_t t, j, i
    
    cdef float planeMean1, varStd, varSkew, varKurt, stdSum, skewSum, kurtSum
    cdef int recs = tMax*jMax*iMax
    
    cdef np.ndarray[DTYPE32_t, ndim=1] varPert  = np.zeros(recs,dtype=DTYPE32)
    cdef np.ndarray[DTYPE32_t, ndim=1] varList  = np.zeros(recs,dtype=DTYPE32)
        
    varList = varArray1.reshape(recs)
    
    #################################################
    # Calculate planar means for each height/time   #
    #################################################
    for j in range(0,recs):
        planeMean1 = planeMean1 + varList[j] / (recs)
    
    for j in range(0,recs):
        varPert[j] = varList[j] - planeMean1
        stdSum = stdSum + (varPert[j]*varPert[j]) / (recs)
    #varStd = np.sqrt(stdSum)
    varStd = stdSum
    
    for j in range(0,recs):
        skewSum = skewSum + (varPert[j]*varPert[j]*varPert[j]) / ((recs-1)*varStd*varStd*varStd)
        kurtSum = kurtSum + (varPert[j]*varPert[j]*varPert[j]*varPert[j]) / ((recs-1)*varStd*varStd*varStd*varStd)
    
    varSkew = skewSum
    varKurt = kurtSum - 3
    return varList, varStd, varSkew, varKurt

###################################################
# Define 1D 1-sided autospectral density function #
###################################################
@cython.cdivision(True)
def spectra1D_meanrecord(np.ndarray[DTYPE64_t, ndim=4] var,float dx):
    
    ################################################
    # Define static types for constants and arrays #
    ################################################
    cdef int tMax = var.shape[0]
    cdef int kMax = var.shape[1]
    cdef int jMax = var.shape[2]
    cdef int iMax = var.shape[3]
    cdef int mMax = iMax/2
    #cdef int base = math.log(iMax,2)
    
    cdef float Summer_mod, Summer_pert
    cdef float summer
    cdef double complex Summer_phi
    cdef Py_ssize_t t, k, j, m, n, b
    
    cdef np.ndarray[DTYPEDC_t, ndim=4] Phi   = np.zeros((tMax,kMax,jMax,mMax+1),dtype=DTYPEDC)
    cdef np.ndarray[DTYPE64_t, ndim=1] mean  = np.zeros(jMax,dtype=DTYPE64)
    #cdef np.ndarray[DTYPE64_t, ndim=1] mList = np.array([2**b for b in range(0,base)],dtype=DTYPE64)
    #cdef int mMax1 = mList.shape[0]
    #cdef int tmp
        
    #for b in range(0,mMax1-2):
        #tmp   = int(mList[b]) + int(mList[b+1])
        #mList = np.insert(mList,-1,tmp)

    #mList = np.sort(mList)
    #mList = np.insert(mList,0,0)
    #mMax1 = mList.shape[0]
    
    cdef np.ndarray[DTYPE64_t, ndim=3] PSD   = np.zeros((tMax,kMax,mMax+1),   dtype=DTYPE64)
    
    #########################################
    # Calculate means of plane and subtract #
    #########################################
    for t in range(0,tMax):
        for k in range(0,kMax):
            for m in range(0,jMax):
                summer = 0
                for n in range(0,iMax):
                    summer = summer + var[t,k,m,n]
                
                mean[m] = summer / (iMax)
            
            for m in range(0,jMax):
                for n in range(0,iMax):
                    var[t,k,m,n] = var[t,k,m,n] - mean[m]
    
    ###########################################################
    # Calculate power spectrum via discrete fourier transform #
    ###########################################################
    for t in range(0,tMax):
        for k in range(0,kMax):
            for j in range(0,jMax):
                Phi[t,k,j,:] = dx * np.fft.fft(var[t,k,j,:])[0:mMax+1]
                    
    ##############################################
    # Calculate the 1-sided autospectral density #
    ##############################################
    for t in range(0,tMax):
        for k in range(kMax):
            for m in range(0,mMax+1):
                Summer_mod = 0
                for j in range(0,jMax):
                    Summer_mod = Summer_mod + cabs(Phi[t,k,j,m])**2
                    # if (mList[m]==0):
#                         l = iMax-1
#                         Summer_mod = Summer_mod + cabs(Phi[t,k,j,l])**2
#                     else:
#                         l = mList[m]-1
#                         Summer_mod = Summer_mod + cabs(Phi[t,k,j,l])**2
                if (m==0 or m==mMax):
                    PSD[t,k,m] = Summer_mod * (1./(jMax*iMax*dx)) / (2*pi)
                else:
                    PSD[t,k,m] = Summer_mod * (2./(jMax*iMax*dx)) / (2*pi)
    return PSD

###################################################
# Define 1D 1-sided autospectral density function #
###################################################
@cython.cdivision(True)
def spectra1D_meanplane(np.ndarray[DTYPE64_t, ndim=4] var,float dx):
    
    ################################################
    # Define static types for constants and arrays #
    ################################################
    cdef int tMax = var.shape[0]
    cdef int kMax = var.shape[1]
    cdef int jMax = var.shape[2]
    cdef int iMax = var.shape[3]
    cdef int mMax = iMax/2
    #cdef int base = math.log(iMax,2)
    
    cdef float Summer_mod, Summer_pert
    cdef float summer, mean
    cdef double complex Summer_phi
    cdef Py_ssize_t t, k, j, m, n, b
    
    cdef np.ndarray[DTYPEDC_t, ndim=4] Phi   = np.zeros((tMax,kMax,jMax,mMax+1),dtype=DTYPEDC)
    #cdef np.ndarray[DTYPE64_t, ndim=1] mList = np.array([2**b for b in range(0,base)],dtype=DTYPE64)
    #cdef int mMax1 = mList.shape[0]
    #cdef int tmp
        
    #for b in range(0,mMax1-2):
        #tmp   = int(mList[b]) + int(mList[b+1])
        #mList = np.insert(mList,-1,tmp)

    #mList = np.sort(mList)
    #mList = np.insert(mList,0,0)
    #mMax1 = mList.shape[0]
    
    cdef np.ndarray[DTYPE64_t, ndim=3] PSD   = np.zeros((tMax,kMax,mMax+1),   dtype=DTYPE64)
    
    #########################################
    # Calculate means of plane and subtract #
    #########################################
    for t in range(0,tMax):
        for k in range(0,kMax):
            summer = 0
            for m in range(0,jMax):
                for n in range(0,iMax):
                    summer = summer + var[t,k,m,n]
            mean = summer / (jMax*iMax)
                
            for m in range(0,jMax):
                for n in range(0,iMax):
                    var[t,k,m,n] = var[t,k,m,n] - mean
    
    ###########################################################
    # Calculate power spectrum via discrete fourier transform #
    ###########################################################
    for t in range(0,tMax):
        for k in range(0,kMax):
            for j in range(0,jMax):
                Phi[t,k,j,:] = dx * np.fft.fft(var[t,k,j,:])[0:mMax+1]
                    
    ##############################################
    # Calculate the 1-sided autospectral density #
    ##############################################
    for t in range(0,tMax):
        for k in range(kMax):
            for m in range(0,mMax+1):
                Summer_mod = 0
                for j in range(0,jMax):
                    Summer_mod = Summer_mod + cabs(Phi[t,k,j,m])**2
                    # if (mList[m]==0):
#                         l = iMax-1
#                         Summer_mod = Summer_mod + cabs(Phi[t,k,j,l])**2
#                     else:
#                         l = mList[m]-1
#                         Summer_mod = Summer_mod + cabs(Phi[t,k,j,l])**2
                if (m==0 or m==mMax):
                    PSD[t,k,m] = Summer_mod * (1./(jMax*iMax*dx)) / (2*pi)
                else:
                    PSD[t,k,m] = Summer_mod * (2./(jMax*iMax*dx)) / (2*pi)
    return PSD

###################################################
# Define 1D 1-sided autospectral density function #
###################################################
@cython.cdivision(True)
def spectra1DNoAvg(np.ndarray[DTYPE64_t, ndim=4] var,float dx):
    
    ################################################
    # Define static types for constants and arrays #
    ################################################
    cdef int tMax = var.shape[0]
    cdef int kMax = var.shape[1]
    cdef int jMax = var.shape[2]
    cdef int iMax = var.shape[3]
    cdef int mMax = iMax/2
    #cdef int base = math.log(iMax,2)
    
    cdef float Summer_mod, Summer_pert
    cdef float summer, mean
    cdef double complex Summer_phi
    cdef Py_ssize_t t, k, j, m, n, b
    
    cdef np.ndarray[DTYPEDC_t, ndim=4] Phi   = np.zeros((tMax,kMax,jMax,mMax+1),dtype=DTYPEDC)
    cdef np.ndarray[DTYPE64_t, ndim=4] PSD   = np.zeros((tMax,kMax,jMax,mMax+1),dtype=DTYPE64)
    
    #########################################
    # Calculate means of plane and subtract #
    #########################################
    for t in range(0,tMax):
        for k in range(0,kMax):
            summer = 0
            for m in range(0,jMax):
                for n in range(0,iMax):
                    summer = summer + var[t,k,m,n]
            mean = summer / (iMax*jMax)
            
            for m in range(0,jMax):
                for n in range(0,iMax):
                    var[t,k,m,n] = var[t,k,m,n] - mean
    
    ###########################################################
    # Calculate power spectrum via discrete fourier transform #
    ###########################################################
    for t in range(0,tMax):
        for k in range(0,kMax):
            for j in range(0,jMax):
                Phi[t,k,j,:] = dx * np.fft.fft(var[t,k,j,:])[0:mMax+1]
                    
    ##############################################
    # Calculate the 1-sided autospectral density #
    ##############################################
    for t in range(0,tMax):
        for k in range(kMax):
            for m in range(0,mMax+1):
                Summer_mod = 0
                for j in range(0,jMax):
                    Summer_mod = cabs(Phi[t,k,j,m])**2
                    # if (mList[m]==0):
#                         l = iMax-1
#                         Summer_mod = Summer_mod + cabs(Phi[t,k,j,l])**2
#                     else:
#                         l = mList[m]-1
#                         Summer_mod = Summer_mod + cabs(Phi[t,k,j,l])**2
                    if (m==0 or m==mMax):
                        PSD[t,k,j,m] = Summer_mod * (1./(iMax*dx)) / (2*pi)
                    else:
                        PSD[t,k,j,m] = Summer_mod * (2./(iMax*dx)) / (2*pi)
    return PSD

###################################################
# Define 1D 1-sided autospectral density function #
# with variable intervals                         #
###################################################
@cython.cdivision(True)
def spectra1DVar(np.ndarray[DTYPE64_t, ndim=1] var, float dx):
    
    ################################################
    # Define static types for constants and arrays #
    ################################################
    cdef int iMax = var.shape[0]
    cdef int mMax = iMax/2

    cdef float summer,mean
    cdef Py_ssize_t m, n
    
    cdef np.ndarray[DTYPEDC_t, ndim=1] Phi   = np.zeros((iMax),   dtype=DTYPEDC)
    cdef np.ndarray[DTYPE64_t, ndim=1] PSD   = np.zeros((mMax+1), dtype=DTYPE64)
    
    #########################################
    # Calculate means of plane and subtract #
    #########################################
    summer = 0
    for n in range(0,iMax):
        summer = summer + var[n]
    mean = summer / (iMax)
    
    for n in range(0,iMax):
        var[n] = var[n] - mean
    
    ###########################################################
    # Calculate power spectrum via discrete fourier transform #
    ###########################################################
    Phi = dx * np.fft.fft(var)[0:mMax+1]
    
    ##############################################
    # Calculate the 1-sided autospectral density #
    ##############################################
    for m in range(0,mMax+1):
        if (m==0 or m==mMax):
            PSD[m] = cabs(Phi[m])**2 * (1./(iMax*dx)) / (2*pi)
        else:
            PSD[m] = cabs(Phi[m])**2 * (2./(iMax*dx)) / (2*pi)
    return PSD

###################################################
# Define 2D 1-sided autospectral density function #
###################################################
@cython.cdivision(True)
def spectra2D(np.ndarray[DTYPE64_t, ndim=4] var,float dx):
    
    ################################################
    # Define static types for constants and arrays #
    ################################################
    cdef float pi = M_PI
    cdef int tMax = var.shape[0]
    cdef int kMax = var.shape[1]
    cdef int jMax = var.shape[2]
    cdef int iMax = var.shape[3]
    cdef int mMax = iMax/2
    cdef double complex Summer_phi
    cdef Py_ssize_t t, k, j, m, n
    cdef float summer, mean
    
    cdef np.ndarray[DTYPEDC_t, ndim=4] Phi   = np.zeros((tMax,kMax,jMax,iMax),dtype=DTYPEDC)
    cdef np.ndarray[DTYPE64_t, ndim=4] PSD   = np.zeros((tMax,kMax,jMax,iMax),dtype=DTYPE64)
    #cdef np.ndarray[DTYPEDC_t, ndim=1] iList = np.exp(-2*pi*i* (np.arange(0,iMax,1.0)/iMax + np.arange(0,jMax,1.0)/jMax)) * dx * dx
    
    #########################################
    # Calculate means of plane and subtract #
    #########################################
    for t in range(0,tMax):
        for k in range(0,kMax):
            summer = 0
            for m in range(0,jMax):
                for n in range(0,iMax):
                    summer = summer + var[t,k,m,n]
            mean = summer / (iMax*jMax)
            
            for m in range(0,jMax):
                for n in range(0,iMax):
                    var[t,k,m,n] = var[t,k,m,n] - mean
    
    ###########################################################
    # Calculate power spectrum via discrete fourier transform #
    ###########################################################
    for t in range(0,tMax):
        for k in range(0,kMax):
            Phi[t,k,:,:] = dx * dx * np.fft.fft2(var[t,k,:,:])
            for m in range(0,jMax):
                for n in range(0,iMax):
                    PSD[t,k,m,n] = cabs(Phi[t,k,m,n])**2 / (jMax*iMax*dx*dx) / (2*pi)**2
    return PSD
