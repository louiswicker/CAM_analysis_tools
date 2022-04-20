
!>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
subroutine SIM_W_1D(dt, km, rgas, gama, gm2, cp2, kappa, pe, dm2, pm2, pem, w2, dz2, pt2, ws, p_fac)
                        
   integer,       intent(in)  :: km
   real(kind=4),  intent(in)  :: dt, rgas, gama, kappa, p_fac
   real(kind=4),  intent(in), dimension(km):: dm2, pt2, pm2, gm2, cp2
   real(kind=4),  intent(in ) ::  ws
   real(kind=4),  intent(in ), dimension(km+1):: pem
   real(kind=4),  intent(inout) ::  pe(km+1), dz2(km), w2(km)
   
   !f2py intent(overwrite) :: pe, dz2, w2

! Local variables

   real(kind=4), dimension(km  ):: Ak, Bk, Rk, Ck, g_rat, bb, dd, aa, cc, dz1, w1
   real(kind=4), dimension(km+1):: pp, gam, wE, dm2e, wES
   real  rdt, capa1, bet

   rdt   = 1. / dt
   capa1 = kappa - 1.

! Compute non-hydrostatic pert pressure

   do k = 1,km
    
     pe(k)   = exp(gama*log(-dm2(k)/dz2(k)*rgas*pt2(k))) - pm2(k)        
     dm2e(k) = dm2(k)
    
     w1(k)   = w2(k)
     dz1(k)  = dz2(k)
    
   enddo

   dm2e(km+1) = dm2(km)
    
! Set up tridiagonal coeffs for spline interpolation of w to grid edges. (Extra copy of bb for end calcs)

    do k = 1,km-1
       
      g_rat(k) = dm2e(k)/dm2e(k+1)
    
      aa(k)    = 1.0
      bb(k)    = 2.*(1.+g_rat(k))
      cc(k)    = g_rat(k)
      dd(k)    = 3.*(w1(k) + cc(k)*w1(k+1))

    enddo
    
! Boundary conditions for wE = 0 at top

    bet    = bb(1)
    wE(1)  = 0.0
    wE(2)  = dd(1) / bet
    bb(km) = 2.
    dd(km) = 3.*w1(km)
       
! Forward calculation of tri-diagonal system

    do k = 2, km
    
      gam(k)  =  g_rat(k-1) / bet
      bet     =  bb(k) - gam(k)
      wE(k+1) = (dd(k) - aa(k)*wE(k) ) / bet
         
    enddo

! Boundary conditions for wE = [ws(i) at ground]  
  
      wE(km+1) = 0.0

! Do the back substition, result is wE on zone edges.

    do k = km, 2, -1

      wE(k) = wE(k) - gam(k)*wE(k+1)
        
    enddo
    
! Compute cell centered tridiagonal coefficients 

    do k = 1, km
    
      aa(k) = dt * gama * (pm2(k)+pe(k)) / dz1(k) 
        
    enddo

! Compute edge centered tridiagonal coefficients and RHS

    do k = 2, km

      cc(k) = 2.0 * dt / (dm2(k) + dm2(k-1))
        
      Bk(k) = 1.0 - cc(k) * (aa(k-1) + aa(k))
    
      Ak(k) = aa(k-1)*cc(k)
        
      Ck(k) = aa(k  )*cc(k)
    
      Rk(k) = wE(k) + cc(k) * (pe(k) - pe(k-1))

    enddo    

! Boundary value calc for forward tri-diagonal solution

   Ck(1)  = 0.0

   Rk(2)  = Rk(2)  - 0.0 * Ck(2)    ! this includes the lower bc for w zero here
    
   Rk(km) = Rk(km) - 0.0 * Ck(km)   ! this includes the lower bc for ws  (zero here)
  
! Forward sweep.

    bet   = Bk(2)

    wE(1) = 0.0
        
    do k = 2, km

      gam(k) = Ck(k-1) / bet
        
      bet    = Bk(k) - Ak(k) * gam(k)

      wE(k)  = (Rk(k) - Ak(k)*wE(k-1) ) / bet

    enddo
    
    wE(km+1)  = 0.0
    gam(km+1) = 0.0
            
! Back substitution for solution (wE = 0 at k=1)

    do k = km-1, 2, -1
      
      wE(k) = wE(k) - gam(k+1)*wE(k+1)
            
    enddo
        
! Solve for new perturbation pressure

    do k = 1, km

      pp(k)  = pe(k) - aa(k) * (wE(k+1) - wE(k))

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
    
! Retrieve edge pressures

   pe(1) = 0.0

    do k = 1,km
      
      !pe(k+1) = pe(k) + dm2(k)*(w2(k) - w1(k))*rdt
      pe(k+1) = pe(k) + pp(k)
        
    enddo
    
 end subroutine SIM_W_1D
