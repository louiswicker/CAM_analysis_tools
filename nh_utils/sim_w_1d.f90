
!>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
subroutine SIM_W_1D(dt, km, rgas, gama, gm2, cp2, kappa, pe, dm2, pm2, pem, w2, dz2, pt2, ws, p_fac, pe_compute)
                        
   integer,       intent(in)  :: km, pe_compute
   real(kind=4),  intent(in)  :: dt, rgas, gama, kappa, p_fac
   real(kind=4),  intent(in), dimension(km):: dm2, pt2, pm2, gm2, cp2
   real(kind=4),  intent(in ) :: ws
   real(kind=4),  intent(in ), dimension(km+1):: pem
   real(kind=4),  intent(inout) :: pe(km+1), dz2(km), w2(km)
   
   !f2py intent(overwrite) :: pe, dz2, w2

! Local variables

   real(kind=4), dimension(km  ):: Ak, Bk, Rk, Ck, g_rat, bb, dd, aa, cc, dz1, w1
   real(kind=4), dimension(km+1):: pp, gam, wE, dm2e, wES
   real  rdt, capa1, bet, r3
    
   real :: dwup, dwcn, dwdn, dmtot, wup, wdn, wcn, dzup, dzdn, dzcn, ddmup, ddmcn, ddmdn

   rdt   = 1. / dt
   capa1 = kappa - 1.
   r3    = 1./3.

! Compute non-hydrostatic pert pressure

   do k = 1,km
    
     pp(k)   = exp(gama*log(-dm2(k)/dz2(k)*rgas*pt2(k))) - pm2(k)        
     dm2e(k) = dm2(k)
    
     w1(k)   = w2(k)
     dz1(k)  = dz2(k)
    
   enddo

   dm2e(km+1) = dm2(km)
    
! Set up tridiagonal coeffs for spline interpolation of w to grid edges. (Extra copy of bb for end calcs)

    do k = 2,km
       
      g_rat(k) = dm2e(k-1)/dm2e(k)
    
      aa(k)    = 1.0
      bb(k)    = 2.*(1.+g_rat(k))
      cc(k)    = g_rat(k)
      dd(k)    = 3.*(w1(k) + cc(k)*w1(k+1))

    enddo
    
! Boundary conditions for von Neuman at the top

    bb(1) = 2.0
    cc(1) = 1.0
    dd(1) = 3.0*w1(1)

! Boundary conditions at the bottom...

    bb(km) = 2.
    dd(km) = 3.*w1(km)
    wE(km+1) = 0.0
    
! Forward calculation of tri-diagonal system  # VBA algorithm from Wikipedia

    do k = 2, km
    
      bet   = aa(k) / bb(k-1)
      bb(k) = bb(k) - bet * cc(k-1)
      dd(k) = dd(k) - bet * dd(k-1)
         
    enddo

! Solve for the last value of matrix
        
      wE(km)   = dd(km) / bb(km)

! Do the back substition, result is wE on zone edges.

    do k = km-1, 2, -1

      wE(k) = (dd(k) - cc(k) * wE(k+1)) / bb(k)
        
    enddo
    
    wE(1) = wE(2)
    
! Store off the edge values

    do k = 1, km+1
    
      wES(k) = wE(k)
        
    enddo 
    
! New solver create nh-pe values for later update.

! Create PE's from original code....

    do k=1,km-1
      g_rat(k) = dm2(k)/dm2(k+1)
         bb(k) = 2.*(1.+g_rat(k))
         dd(k) = 3.*(pp(k) + g_rat(k)*pp(k+1))
    enddo

    bet = bb(1)
    pp(1) = 0.
    pp(2) = dd(1) / bet
    bb(km) = 2.
    dd(km) = 3.*pp(km)

! Forward calculation of tri-diagonal system

    do k=2,km
      gam(k) =  g_rat(k-1) / bet
      bet    =  bb(k) - gam(k)
      pe(k+1) = (dd(k) - pe(k) ) / bet
    enddo

! Do the back substition, result is pp on zone edges.

    do k=km, 2, -1
       pe(k) = pe(k) - gam(k)*pe(k+1)
    enddo
    
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Now start the implicit solver

! Compute cell centered tridiagonal coefficients 

    do k = 1, km
    
      Ak(k) = dt * gama * (pm2(k)+pp(k)) / dz1(k) 
        
    enddo

! Compute edge centered tridiagonal coefficients and RHS

    do k = 2, km

      Ck(k) = 2.0 * dt / (dm2(k) + dm2(k-1))
    
      aa(k) = Ak(k-1)*Ck(k)
        
      cc(k) = Ak(k  )*Ck(k)
    
      dd(k) = wE(k) + Ck(k) * (pp(k) - pp(k-1))
        
      bb(k) = 1.0 - cc(k) - aa(k)

    enddo    

! Boundary value calc for forward tri-diagonal solution

! von Neuman conditions

   bb(2) = 1 - aa(2)
    
   dd(km) = dd(km) - 0.0 * Ck(km)   ! this includes the lower bc for ws  (zero here)
  
! Forward calculation of tri-diagonal system  # VBA algorithm from Wikipedia

    do k = 3, km
    
      bet   = aa(k) / bb(k-1)
      bb(k) = bb(k) - bet * cc(k-1)
      dd(k) = dd(k) - bet * dd(k-1)
         
    enddo

! Solve for the last value of matrix
        
      wE(km)   = dd(km) / bb(km)
        
      wE(km+1) = 0.0
        
! Do the back substition, result is wE on zone edges.

    do k = km-1, 2, -1

      wE(k) = (dd(k) - cc(k) * wE(k+1)) / bb(k)
        
    enddo
    
    wE(1) = wE(2)
        
! Solve for new perturbation pressure at the cell centers.

    do k = 1, km

      pp(k)  = pp(k) - Ak(k) * (wE(k+1) - wE(k))

    enddo
    
! Use new pp at cell centers to get new dz's

    do k = 1, km
    
      dz2(k) = -dm2(k)*rgas*pt2(k)*exp(capa1*log(pp(k)+pm2(k)))

    enddo
    
! Need to generate new w at cell centers...use the time tendency of dz

    w2(km) = ( wE(km) + 2.*wE(km+1) )*r3
    
    do k = km-1,1,-1

     !w2(k) = 0.5*(wE(k) + wE(k+1))
        
     w2(k) = (wE(k) + bb(k)*wE(k+1) + g_rat(k)*wE(k+2))*r3 - g_rat(k)*w2(k+1)
        
    enddo
    
    pe(1) = 0.0
    
    if( pe_compute .eq. 1 ) THEN
    
      do k=1,km
          pe(k+1) = pe(k) + dm2(k)*(w2(k)-w1(k))*rdt
      enddo

    ENDIF 
    
    IF( pe_compute .eq. 2 ) THEN
    
! Edge code from original nh_utils/SIM1

        do k=1,km-1
          g_rat(k) = dm2(k)/dm2(k+1)
             bb(k) = 2.*(1.+g_rat(k))
             dd(k) = 3.*(pp(k) + g_rat(k)*pp(k+1))
        enddo

        bet = bb(1)
        pp(1) = 0.
        pp(2) = dd(1) / bet
        bb(km) = 2.
        dd(km) = 3.*pp(km)

! Forward calculation of tri-diagonal system

        do k=2,km
          gam(k) =  g_rat(k-1) / bet
          bet    =  bb(k) - gam(k)
          pe(k+1) = (dd(k) - pe(k) ) / bet
        enddo

! Do the back substition, result is pp on zone edges.

        do k=km, 2, -1
           pe(k) = pe(k) - gam(k)*pe(k+1)
        enddo
            
    ENDIF

   IF( pe_compute .eq. 3 ) THEN
    
! Solve for new perturbation pressure at the cell EDGES

     do k = 2, km

       pe(k)  = pe(k) - 0.5*(Ak(k)+Ak(k-1)) * (w2(k) - w2(k-1))

     enddo
            
    ENDIF
    
end subroutine SIM_W_1D
