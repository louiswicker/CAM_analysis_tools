!     -*- f90 -*-
!     This file is autogenerated with f2py (version:1.21.5)
!     It contains Fortran 90 wrappers to fortran functions.

      
      subroutine f2pyinitcmpref_mod(f2pysetupfunc)
      use cmpref_mod, only : calcrefl10cm
      use cmpref_mod, only : calcrefl10cm1d
      use cmpref_mod, only : rayleighsoakwetgraupel
      use cmpref_mod, only : get_m_mix_nested
      use cmpref_mod, only : get_m_mix
      use cmpref_mod, only : mcomplexmaxwellgarnett
      use cmpref_mod, only : make_rainnumber
      external f2pysetupfunc
      call f2pysetupfunc(calcrefl10cm,calcrefl10cm1d,rayleighsoakwetgrau&
     &pel,get_m_mix_nested,get_m_mix,mcomplexmaxwellgarnett,make_rainnum&
     &ber)
      end subroutine f2pyinitcmpref_mod

      
      subroutine f2pyinitshare_mod(f2pysetupfunc)
      use share_mod, only : nrbins
      use share_mod, only : xxDx
      use share_mod, only : xxDs
      use share_mod, only : xdts
      use share_mod, only : xxDg
      use share_mod, only : xdtg
      use share_mod, only : lamdaradar
      use share_mod, only : Kw
      use share_mod, only : PI5
      use share_mod, only : lamda4
      use share_mod, only : mw0
      use share_mod, only : mi0
      use share_mod, only : simpson
      use share_mod, only : basis
      use share_mod, only : xcre
      use share_mod, only : xcse
      use share_mod, only : xcge
      use share_mod, only : xcrg
      use share_mod, only : xcsg
      use share_mod, only : xcgg
      use share_mod, only : xamr
      use share_mod, only : xbmr
      use share_mod, only : xmur
      use share_mod, only : xobmr
      use share_mod, only : xams
      use share_mod, only : xbms
      use share_mod, only : xmus
      use share_mod, only : xoams
      use share_mod, only : xobms
      use share_mod, only : xocms
      use share_mod, only : xamg
      use share_mod, only : xbmg
      use share_mod, only : xmug
      use share_mod, only : xoamg
      use share_mod, only : xobmg
      use share_mod, only : xocmg
      use share_mod, only : xorg2
      use share_mod, only : xosg2
      use share_mod, only : xogg2
      use share_mod, only : PI
      use share_mod, only : slen
      use share_mod, only : mixingrulestrings
      use share_mod, only : matrixstrings
      use share_mod, only : inclusionstrings
      use share_mod, only : hoststrings
      use share_mod, only : hostmatrixstrings
      use share_mod, only : hostinclusionstrings
      use share_mod, only : mixingrulestringg
      use share_mod, only : matrixstringg
      use share_mod, only : inclusionstringg
      use share_mod, only : hoststringg
      use share_mod, only : hostmatrixstringg
      use share_mod, only : hostinclusionstringg
      use share_mod, only : meltoutsides
      use share_mod, only : meltoutsideg
      use share_mod, only : radar_init
      use share_mod, only : m_complex_water_ray
      use share_mod, only : m_complex_ice_maetzler
      use share_mod, only : wgamma
      external f2pysetupfunc
      call f2pysetupfunc(nrbins,xxDx,xxDs,xdts,xxDg,xdtg,lamdaradar,Kw,P&
     &I5,lamda4,mw0,mi0,simpson,basis,xcre,xcse,xcge,xcrg,xcsg,xcgg,xamr&
     &,xbmr,xmur,xobmr,xams,xbms,xmus,xoams,xobms,xocms,xamg,xbmg,xmug,x&
     &oamg,xobmg,xocmg,xorg2,xosg2,xogg2,PI,slen,mixingrulestrings,matri&
     &xstrings,inclusionstrings,hoststrings,hostmatrixstrings,hostinclus&
     &ionstrings,mixingrulestringg,matrixstringg,inclusionstringg,hostst&
     &ringg,hostmatrixstringg,hostinclusionstringg,meltoutsides,meltouts&
     &ideg,radar_init,m_complex_water_ray,m_complex_ice_maetzler,wgamma)
      end subroutine f2pyinitshare_mod


