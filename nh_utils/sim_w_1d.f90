
!>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
subroutine SIM_W_1D(dt, km, rgas, gama, gm2, cp2, kappa, pe, dm2, pm2, pem, w2, dz2, pt2, ws, p_fac)
                        
   integer,       intent(in)  :: km
   real(kind=4),  intent(in)  :: dt, rgas, gama, kappa, p_fac
   real(kind=4),  intent(in), dimension(km):: dm2, pt2, pm2, gm2, cp2
   real(kind=4),  intent(in ) :: ws
   real(kind=4),  intent(in ), dimension(km+1):: pem
   real(kind=4),  intent(inout) :: pe(km+1), dz2(km), w2(km)
   
   !f2py intent(overwrite) :: pe, dz2, w2

! Local variables

   real(kind=4), dimension(km  ):: Ak, Bk, Rk, Ck, g_rat, bb, dd, aa, cc, dz1, w1
   real(kind=4), dimension(km+1):: pp, gam, wE, dm2e, wES
   real  rdt, capa1, bet

   rdt   = 1. / dt
   capa1 = kappa - 1.

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
    
! Boundary conditions for wE = 0 at top

    wE(1)  = 0.0
    wE(km+1) = 0.0

    bb(km) = 2.
    dd(km) = 3.*w1(km)
       
! Forward calculation of tri-diagonal system  # VBA algorithm from Wikipedia

    do k = 3, km
    
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

   dd(2)  = dd(2)  - 0.0 * Ck(2)    ! this includes the lower bc for w zero here
    
   dd(km) = dd(km) - 0.0 * Ck(km)   ! this includes the lower bc for ws  (zero here)
  
! Forward calculation of tri-diagonal system  # VBA algorithm from Wikipedia

    do k = 3, km
    
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
        
! Solve for new perturbation pressure

    do k = 1, km

      pp(k)  = pp(k) - Ak(k) * (wE(k+1) - wE(k))

    enddo
    
! Use new pp at cell centers to get new dz's

    do k = 1, km
    
      dz2(k) = -dm2(k)*rgas*pt2(k)*exp(capa1*log(pp(k)+pm2(k)))

    enddo
    
! Need to generate new w at cell centers...use the time tendency of dz

    do k = 1,km

     !w2(k) = w1(k) - rdt * (dz2(k) - dz1(k))
     w2(k) = 0.5*(wE(k) + wE(k+1))
        
    enddo
    
! Set up tridiagonal coeffs for spline interpolation of pe to grid edges.

    do k = 2,km
       
      g_rat(k) = dm2e(k-1)/dm2e(k)
    
      aa(k)    = 1.0
      bb(k)    = 2.*(1.+g_rat(k))
      cc(k)    = g_rat(k)
      dd(k)    = 3.*(pp(k) + cc(k)*pp(k+1))

    enddo
    
! Boundary conditions for wE = 0 at top

    g_rat(km+1) = 1.0
   
    aa(km+1) = 1.0
    bb(km+1) = 4.0
    cc(km+1) = 1.0
    
    pe(1) = 0.0
    
!    bb(km+1) = 2.
    dd(km+1) = 3.*pp(km)
       
! Forward calculation of tri-diagonal system  # VBA algorithm from Wikipedia

    do k = 3, km+1
    
      bet   = aa(k) / bb(k-1)
      bb(k) = bb(k) - bet * cc(k-1)
      dd(k) = dd(k) - bet * dd(k-1)
         
    enddo

! Solve for the last value of matrix
        
      pe(km)   = dd(km) / bb(km)

! Do the back substition, result is wE on zone edges.

    do k = km-1, 2, -1

      pe(k) = (dd(k) - cc(k) * pe(k+1)) / bb(k)

    enddo
        
    
! Retrieve edge pressures

!  pe(1) = 0.0

 !   do k = 1,km
      
 !     
 !   enddo
    
 end subroutine SIM_W_1D
