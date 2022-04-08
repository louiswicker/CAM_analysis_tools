module fv_mapz_mod


  implicit none
  real, parameter:: consv_min= 0.001         !< below which no correction applies
  real, parameter:: t_min= 184.              !< below which applies stricter constraint
  real, parameter:: r3 = 1./3., r23 = 2./3., r12 = 1./12.
 ! real, parameter:: cv_vap = 3.*rvgas        !< 1384.5
 ! real, parameter:: cv_air =  cp_air - rdgas !< = rdgas * (7/2-1) = 2.5*rdgas=717.68
! real, parameter:: c_ice = 2106.            !< heat capacity of ice at 0.C
  real, parameter:: c_ice = 1972.            !< heat capacity of ice at -15.C
  real, parameter:: c_liq = 4.1855e+3        !< GFS: heat capacity of water at 0C
! real, parameter:: c_liq = 4218.            !< ECMWF-IFS
 ! real, parameter:: cp_vap = cp_vapor        !< 1846.
  real, parameter:: tice = 273.16

  real, parameter :: w_max = 60.
  real, parameter :: w_min = -30.
  logical, parameter :: w_limiter = .false. ! doesn't work so well??

  real(kind=4) :: E_Flux = 0.

contains
 subroutine cs_profile(qs, a4, delp, km, iv, kord)
! Optimized vertical profile reconstruction:
! Latest: Apr 2008 S.-J. Lin, NOAA/GFDL
 integer, intent(in):: km      !< vertical dimension
 integer, intent(in):: iv      !< iv =-1: winds
                               !< iv = 0: positive definite scalars
                               !< iv = 1: others
 integer, intent(in):: kord
 real, intent(in)   ::   qs
 real, intent(in)   :: delp(km)     !< layer pressure thickness
 real, intent(inout):: a4(4,km)     !< Interpolated values
!-----------------------------------------------------------------------
 logical, dimension(km):: extm, ext5, ext6
 real  gam(km)
 real    q(km+1)
 real   d4
 real   bet, a_bot, grat
 real   pmp_1, lac_1, pmp_2, lac_2, x0, x1
 integer i, k, im

 if ( iv .eq. -2 ) then
    gam(2) = 0.5
    q(1) = 1.5*a4(1,1)
      do k=2,km-1
                  grat = delp(k-1) / delp(k)
                   bet =  2. + grat + grat - gam(k)
                q(k) = (3.*(a4(1,k-1)+a4(1,k)) - q(k-1))/bet
            gam(k+1) = grat / bet
      enddo
            grat = delp(km-1) / delp(km)
         q(km) = (3.*(a4(1,km-1)+a4(1,km)) - grat*qs - q(km-1)) /  &
                   (2. + grat + grat - gam(km))
         q(km+1) = qs
      do k=km-1,1,-1
           q(k) = q(k) - gam(k+1)*q(k+1)
      enddo
 else
         grat = delp(2) / delp(1)   ! grid ratio
          bet = grat*(grat+0.5)
       q(1) = ( (grat+grat)*(grat+1.)*a4(1,1) + a4(1,2) ) / bet
     gam(1) = ( 1. + grat*(grat+1.5) ) / bet

  do k=2,km
           d4 = delp(k-1) / delp(k)
             bet =  2. + d4 + d4 - gam(k-1)
          q(k) = ( 3.*(a4(1,k-1)+d4*a4(1,k)) - q(k-1) )/bet
        gam(k) = d4 / bet
  enddo

         a_bot = 1. + d4*(d4+1.5)
     q(km+1) = (2.*d4*(d4+1.)*a4(1,km)+a4(1,km-1)-a_bot*q(km))  &
               / ( d4*(d4+0.5) - a_bot*gam(km) )

  do k=km,1,-1
        q(k) = q(k) - gam(k)*q(k+1)
  enddo
 endif


!------------------
! Apply constraints


! Apply *large-scale* constraints

     q(2) = min( q(2), max(a4(1,1), a4(1,2)) )
     q(2) = max( q(2), min(a4(1,1), a4(1,2)) )

  do k=2,km
        gam(k) = a4(1,k) - a4(1,k-1)
  enddo

! Interior:
  do k=3,km-1
        if ( gam(k-1)*gam(k+1)>0. ) then
! Apply large-scale constraint to ALL fields if not local max/min
             q(k) = min( q(k), max(a4(1,k-1),a4(1,k)) )
             q(k) = max( q(k), min(a4(1,k-1),a4(1,k)) )
        else
          if ( gam(k-1) > 0. ) then
! There exists a local max
               q(k) = max(q(k), min(a4(1,k-1),a4(1,k)))
          else
! There exists a local min
                 q(k) = min(q(k), max(a4(1,k-1),a4(1,k)))
               if ( iv==0 ) q(k) = max(0., q(k))
          endif
        endif
  enddo

! Bottom:
     q(km) = min( q(km), max(a4(1,km-1), a4(1,km)) )
     q(km) = max( q(km), min(a4(1,km-1), a4(1,km)) )

  do k=1,km
        a4(2,k) = q(k  )
        a4(3,k) = q(k+1)
  enddo

  do k=1,km
     if ( k==1 .or. k==km ) then
          extm(k) = (a4(2,k)-a4(1,k)) * (a4(3,k)-a4(1,k)) > 0.
     else
          extm(k) = gam(k)*gam(k+1) < 0.
     endif
     if ( abs(kord) > 9 ) then
          x0 = 2.*a4(1,k) - (a4(2,k)+a4(3,k))
          x1 = abs(a4(2,k)-a4(3,k))
          a4(4,k) = 3.*x0
          ext5(k) = abs(x0) > x1
          ext6(k) = abs(a4(4,k)) > x1
     endif
  enddo

!---------------------------
! Apply subgrid constraints:
!---------------------------
! f(s) = AL + s*[(AR-AL) + A6*(1-s)]         ( 0 <= s  <= 1 )
! Top 2 and bottom 2 layers always use monotonic mapping

  if ( iv==0 ) then
        a4(2,1) = max(0., a4(2,1))
  elseif ( iv==-1 ) then
         if ( a4(2,1)*a4(1,1) <= 0. ) a4(2,1) = 0.
  elseif ( iv==2 ) then
        a4(2,1) = a4(1,1)
        a4(3,1) = a4(1,1)
        a4(4,1) = 0.
  endif

  if ( iv/=2 ) then
        a4(4,1) = 3.*(2.*a4(1,1) - (a4(2,1)+a4(3,1)))
     call cs_limiters(im, extm(1), a4(1,1), 1)
  endif

! k=2
      a4(4,2) = 3.*(2.*a4(1,2) - (a4(2,2)+a4(3,2)))
   call cs_limiters(im, extm(2), a4(1,2), 2)

!-------------------------------------
! Huynh's 2nd constraint for interior:
!-------------------------------------
  do k=3,km-2
     if ( abs(kord)<9 ) then
! Left  edges
          pmp_1 = a4(1,k) - 2.*gam(k+1)
          lac_1 = pmp_1 + 1.5*gam(k+2)
          a4(2,k) = min(max(a4(2,k), min(a4(1,k), pmp_1, lac_1)),   &
                                         max(a4(1,k), pmp_1, lac_1) )
! Right edges
          pmp_2 = a4(1,k) + 2.*gam(k)
          lac_2 = pmp_2 - 1.5*gam(k-1)
          a4(3,k) = min(max(a4(3,k), min(a4(1,k), pmp_2, lac_2)),    &
                                         max(a4(1,k), pmp_2, lac_2) )

          a4(4,k) = 3.*(2.*a4(1,k) - (a4(2,k)+a4(3,k)))

     elseif ( abs(kord)==9 ) then
          if ( extm(k) .and. extm(k-1) ) then  ! c90_mp122
! grid-scale 2-delta-z wave detected
               a4(2,k) = a4(1,k)
               a4(3,k) = a4(1,k)
               a4(4,k) = 0.
          else if ( extm(k) .and. extm(k+1) ) then  ! c90_mp122
! grid-scale 2-delta-z wave detected
               a4(2,k) = a4(1,k)
               a4(3,k) = a4(1,k)
               a4(4,k) = 0.
          else
            a4(4,k) = 6.*a4(1,k) - 3.*(a4(2,k)+a4(3,k))
! Check within the smooth region if subgrid profile is non-monotonic
            if( abs(a4(4,k)) > abs(a4(2,k)-a4(3,k)) ) then
                  pmp_1 = a4(1,k) - 2.*gam(k+1)
                  lac_1 = pmp_1 + 1.5*gam(k+2)
              a4(2,k) = min(max(a4(2,k), min(a4(1,k), pmp_1, lac_1)),  &
                                             max(a4(1,k), pmp_1, lac_1) )
                  pmp_2 = a4(1,k) + 2.*gam(k)
                  lac_2 = pmp_2 - 1.5*gam(k-1)
              a4(3,k) = min(max(a4(3,k), min(a4(1,k), pmp_2, lac_2)),  &
                                             max(a4(1,k), pmp_2, lac_2) )
              a4(4,k) = 6.*a4(1,k) - 3.*(a4(2,k)+a4(3,k))
            endif
          endif
     endif

! Additional constraint to ensure positivity
     if ( iv==0 ) call cs_limiters(im, extm(k), a4(1,k), 0)

  enddo      ! k-loop

!----------------------------------
! Bottom layer subgrid constraints:
!----------------------------------
  if ( iv==0 ) then
        a4(3,km) = max(0., a4(3,km))
  elseif ( iv .eq. -1 ) then
         if ( a4(3,km)*a4(1,km) <= 0. )  a4(3,km) = 0.
  endif

  do k=km-1,km
        a4(4,k) = 3.*(2.*a4(1,k) - (a4(2,k)+a4(3,k)))
     if(k==(km-1)) call cs_limiters(im, extm(k), a4(1,k), 2)
     if(k== km   ) call cs_limiters(im, extm(k), a4(1,k), 1)
  enddo

 end subroutine cs_profile


 subroutine cs_limiters(im, extm, a4, iv)
 integer, intent(in) :: im
 integer, intent(in) :: iv
 logical, intent(in) :: extm(im)
 real , intent(inout) :: a4(4,im)   !< PPM array
! LOCAL VARIABLES:
 real  da1, da2, a6da
 integer i

 if ( iv==0 ) then
! Positive definite constraint
    do i=1,im
    if( a4(1,i)<=0.) then
        a4(2,i) = a4(1,i)
        a4(3,i) = a4(1,i)
        a4(4,i) = 0.
    else
      if( abs(a4(3,i)-a4(2,i)) < -a4(4,i) ) then
         if( (a4(1,i)+0.25*(a4(3,i)-a4(2,i))**2/a4(4,i)+a4(4,i)*r12) < 0. ) then
! local minimum is negative
             if( a4(1,i)<a4(3,i) .and. a4(1,i)<a4(2,i) ) then
                 a4(3,i) = a4(1,i)
                 a4(2,i) = a4(1,i)
                 a4(4,i) = 0.
             elseif( a4(3,i) > a4(2,i) ) then
                 a4(4,i) = 3.*(a4(2,i)-a4(1,i))
                 a4(3,i) = a4(2,i) - a4(4,i)
             else
                 a4(4,i) = 3.*(a4(3,i)-a4(1,i))
                 a4(2,i) = a4(3,i) - a4(4,i)
             endif
         endif
      endif
    endif
    enddo
 elseif ( iv==1 ) then
    do i=1,im
      if( (a4(1,i)-a4(2,i))*(a4(1,i)-a4(3,i))>=0. ) then
         a4(2,i) = a4(1,i)
         a4(3,i) = a4(1,i)
         a4(4,i) = 0.
      else
         da1  = a4(3,i) - a4(2,i)
         da2  = da1**2
         a6da = a4(4,i)*da1
         if(a6da < -da2) then
            a4(4,i) = 3.*(a4(2,i)-a4(1,i))
            a4(3,i) = a4(2,i) - a4(4,i)
         elseif(a6da > da2) then
            a4(4,i) = 3.*(a4(3,i)-a4(1,i))
            a4(2,i) = a4(3,i) - a4(4,i)
         endif
      endif
    enddo
 else
! Standard PPM constraint
    do i=1,im
      if( extm(i) ) then
         a4(2,i) = a4(1,i)
         a4(3,i) = a4(1,i)
         a4(4,i) = 0.
      else
         da1  = a4(3,i) - a4(2,i)
         da2  = da1**2
         a6da = a4(4,i)*da1
         if(a6da < -da2) then
            a4(4,i) = 3.*(a4(2,i)-a4(1,i))
            a4(3,i) = a4(2,i) - a4(4,i)
         elseif(a6da > da2) then
            a4(4,i) = 3.*(a4(3,i)-a4(1,i))
            a4(2,i) = a4(3,i) - a4(4,i)
         endif
      endif
    enddo
 endif
 end subroutine cs_limiters

end module fv_mapz_mod
