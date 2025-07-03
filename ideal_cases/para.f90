! Interpolate using cubic spline p' to pp at cell edges

SUBROUTINE FV3_SPLINE_1D(pe, dm2, km, pp)

  integer, intent(in) :: km
  real,    intent(in) :: pe(km), dm2(km)
  real,   intent(out) :: pp(km+1)

! Local storage
  real, dimension(km) :: g_rat, bb, dd, gam
  real bet
  integer k

     pp(:) = 0.0

     do k=1,km-1
        g_rat(k) = dm2(k)/dm2(k+1)
           bb(k) = 2.*(1.+g_rat(k))
           dd(k) = 3.*(pe(k) + g_rat(k)*pe(k+1))
     enddo

     bet    = bb(1)
     pp(1)  = 0.
     pp(2)  = dd(1) / bet
     bb(km) = 2.
     dd(km) = 3.*pe(km)

     do k=2,km
        gam(k) =  g_rat(k-1) / bet
          bet  =  bb(k) - gam(k)
       pp(k+1) = (dd(k) - pp(k) ) / bet
     enddo

     do k=km, 2, -1
        pp(k) = pp(k) - gam(k)*pp(k+1)
     enddo


   RETURN
   END

