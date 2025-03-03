# -*- coding: utf-8 -*-
"""
Created Tue Apr 08 2020
@author: pat

This is a collection of RGB color values taken from the colorbrewer 2.0 website 
(http://colorbrewer2.org/) and customized color maps used by the WofS web viewer 
"""

#######################################################################################

import numpy as np
import matplotlib
import pylab as P

#######################################################################################

class cb_colors(object):

   #Defines RGB values for select colors from colorbrewer 2.0 (http://colorbrewer2.org/)

   #Input:

   #None

   #Returns:

   #None

#######################

#These colors are single color colortable RGB values from colorbrewer 2.0 (http://colorbrewer2.org/)

   orange1 = (255 / 255., 245 / 255., 235 / 255.)
   orange2 = (254 / 255., 230 / 255., 206 / 255.)
   orange3 = (253 / 255., 208 / 255., 162 / 255.)
   orange4 = (253 / 255., 174 / 255., 107 / 255.)
   orange5 = (253 / 255., 141 / 255., 60 / 255.)
   orange6 = (241 / 255., 105 / 255., 19 / 255.)
   orange7 = (217 / 255., 72 / 255., 1 / 255.)
   orange8 = (166 / 255., 54 / 255., 3 / 255.)
   orange9 = (127 / 255., 39 / 255., 4 / 255.)

   blue1 = (247/255., 251/255., 255/255.)
   blue2 = (222/255., 235/255., 247/255.)
   blue3 = (198/255., 219/255., 239/255.)
   blue4 = (158/255., 202/255., 225/255.)
   blue5 = (107/255., 174/255., 214/255.)
   blue6 = (66/255., 146/255., 198/255.)
   blue7 = (33/255., 113/255., 181/255.)
   blue8 = (8/255., 81/255., 156/255.)
   blue9 = (8/255., 48/255., 107/255.)

   purple1 = (252/255., 251/255., 253/255.)
   purple2 = (239/255., 237/255., 245/255.)
   purple3 = (218/255., 218/255., 235/255.)
   purple4 = (188/255., 189/255., 220/255.)
   purple5 = (158/255., 154/255., 200/255.)
   purple6 = (128/255., 125/255., 186/255.)
   purple7 = (106/255., 81/255., 163/255.)
   purple8 = (84/255., 39/255., 143/255.)
   purple9 = (63/255., 0/255., 125/255.)

   green1 = (247/255., 252/255., 245/255.)
   green2 = (229/255., 245/255., 224/255.)
   green3 = (199/255., 233/255., 192/255.)
   green4 = (161/255., 217/255., 155/255.)
   green5 = (116/255., 196/255., 118/255.)
   green6 = (65/255., 171/255., 93/255.)
   green7 = (35/255., 139/255., 69/255.)
   green8 = (0/255., 109/255., 44/255.)
   green9 = (0/255., 68/255., 27/255.)

   gray1 = (255/255., 255/255., 255/255.)
   gray2 = (240/255., 240/255., 240/255.)
   gray3 = (217/255., 217/255., 217/255.)
   gray4 = (189/255., 189/255., 189/255.)
   gray5 = (150/255., 150/255., 150/255.)
   gray6 = (115/255., 115/255., 115/255.)
   gray7 = (82/255., 82/255., 82/255.)
   gray8 = (37/255., 37/255., 37/255.)
   gray9 = (0/255., 0/255., 0/255.)

   red1 = (255/255., 245/255., 240/255.)
   red2 = (254/255., 224/255., 210/255.)
   red3 = (252/255., 187/255., 161/255.)
   red4 = (252/255., 146/255., 114/255.)
   red5 = (251/255., 106/255., 74/255.)
   red6 = (239/255., 59/255., 44/255.)
   red7 = (203/255., 24/255., 29/255.)
   red8 = (165/255., 15/255., 21/255.)
   red9 = (103/255., 0/255., 13/255.)

### Qualitative colors (pastels):

   q1 = (141/255., 255/255., 199/255.)  #aqua
   q2 = (255/255., 255/255., 179/255.)  #pale yellow
   q3 = (190/255., 186/255., 218/255.)  #lavender
   q4 = (251/255., 128/255., 114/255.)  #pink/orange
   q5 = (128/255., 177/255., 211/255.)  #light blue
   q6 = (253/255., 180/255., 98/255.)   #light orange
   q7 = (179/255., 222/255., 105/255.)  #lime
   q8 = (252/255., 205/255., 229/255.)  #pink
   q9 = (217/255., 217/255., 217/255.)  #light gray
   q10 = (188/255., 128/255., 189/255.) #purple
   q11 = (204/255., 235/255., 197/255.) #pale green
   q12 = (255/255., 237/255., 111/255.) #yellow

### Qualitative colors (bright):

   b1 = (228/255., 26/255., 28/255.)   #red
   b2 = (55/255., 126/255., 184/255.)  #blue
   b3 = (77/255., 175/255., 74/255.)   #green
   b4 = (152/255., 78/255., 163/255.)  #purple
   b5 = (255/255., 127/255., 0/255.)   #orange
   b6 = (255/255., 255/255., 51/255.)  #yellow
   b7 = (166/255., 86/255., 40/255.)   #brown
   b8 = (247/255., 129/255., 191/255.) #pink

### NWS Reflectivity Colors (courtesy MetPy library):

   c5 =  (0.0,                 0.9254901960784314, 0.9254901960784314)
   c10 = (0.00392156862745098, 0.6274509803921569, 0.9647058823529412)
   c15 = (0.0,                 0.0,                0.9647058823529412)
   c20 = (0.0,                 1.0,                0.0)
   c25 = (0.0,                 0.7843137254901961, 0.0)
   c30 = (0.0,                 0.5647058823529412, 0.0)
   c35 = (1.0,                 1.0,                0.0)
   c40 = (0.9058823529411765,  0.7529411764705882, 0.0)
   c45 = (1.0,                 0.5647058823529412, 0.0)
   c50 = (1.0,                 0.0,                0.0)
   c55 = (0.8392156862745098,  0.0,                0.0)
   c60 = (0.7529411764705882,  0.0,                0.0)
   c65 = (1.0,                 0.0,                1.0)
   c70 = (0.6,                 0.3333333333333333, 0.788235294117647)
   c75 = (0.0,                 0.0,                0.0)


### Custom Colormaps:

   cin_cmap = matplotlib.colors.ListedColormap([purple7, purple6, purple5, \
              purple4, blue4, blue3, blue2, blue1])

   wz_cmap = matplotlib.colors.ListedColormap([blue2, blue3, blue4, red2, red3, \
             red4, red5, red6, red7])

   dz_cmap_2 = matplotlib.colors.ListedColormap([blue5, blue3, green3, green5, \
               green7, orange3, orange5, orange7, red7, red8, purple8, purple6])

   dz_cmap = matplotlib.colors.ListedColormap([green5, green4, green3, orange2, \
             orange4, orange6, red6, red4, purple3, purple5])

   nws_dz_cmap = matplotlib.colors.ListedColormap([c20, c25, c30, c35, c40, c45, \
                 c50, c55, c60, c65, c70])

   nws_dz_cmap_clear_air = matplotlib.colors.ListedColormap([c5, c10, c15, c20, \
                           c25, c30, c35, c40, c45, c50, c55, c60, c65, c70])

   wind_svr_cmap = matplotlib.colors.ListedColormap([gray2, gray3, blue1, \
                    blue2, blue3, blue4, blue5, orange3, orange4, \
                    orange5, red4, red5, red6, red7, red8, purple8, \
                    purple7, purple6])

   wind_trop_cmap = matplotlib.colors.ListedColormap([gray2, gray3, blue1, \
                    blue2, blue3, blue4, blue5, orange3, orange4, orange5, \
                    orange6, orange7, red3, red4, red5, red6, red7, red8, red9, \
                    purple8, purple7])

   wind_cmap = matplotlib.colors.ListedColormap([gray1, gray2, gray3, orange2, \
               orange3, orange4, orange5, orange6, red7, red8])

   wz_cmap_extend = matplotlib.colors.ListedColormap([blue2, blue3, blue4, red2, \
                    red3, red4, red5, red6, red7, purple7, purple6, purple5])

   cape_cmap = matplotlib.colors.ListedColormap([blue2, blue3, blue4, orange2, \
               orange3, orange4, orange5, red4, red5, red6, red7, purple7, \
               purple6, purple5])

   td_cmap_ncar = matplotlib.colors.ListedColormap(['#ad598a', '#c589ac', \
                  '#dcb8cd', '#e7cfd1', '#d0a0a4', '#ad5960', '#8b131d', \
                  '#8b4513', '#ad7c59', '#c5a289', '#dcc7b8', '#eeeeee', \
                  '#dddddd', '#bbbbbb', '#e1e1d3', '#e1d5b1', '#ccb77a', \
                  '#ffffe5', '#f7fcb9', '#addd8e', '#41ab5d', '#006837', \
                  '#004529', '#195257', '#4c787c'])

   temp_cmap_ugly = matplotlib.colors.ListedColormap([blue6, blue4, blue2, \
                    green6, green4, green2, orange2, orange4, red5, red7, purple7])

   temp_cmap = matplotlib.colors.ListedColormap([purple4, purple5, purple6, \
               purple7, blue8, blue7, blue6, blue5, blue4, blue3, green7, \
               green6, green5, green4, green3, green2, orange2, orange3, \
               orange4, orange5, red5, red6, red7, red8, purple6, purple5, \
               purple4, purple3])

   blues_cmap = matplotlib.colors.ListedColormap([blue3, blue4, blue5, blue6, \
                blue7])

   oranges_cmap = matplotlib.colors.ListedColormap([orange3, orange4, orange5, \
                  orange6, orange7])

   td_cmap_ugly = matplotlib.colors.ListedColormap([orange3, gray4, gray3, \
                  gray1, green3, green5, green7, blue3, blue5, blue7, purple3])

   td_cmap = matplotlib.colors.ListedColormap([gray6, gray5, gray4, gray3, \
             gray2, gray1, green1, green2, green3, green4, green5, green6, \
             blue3, blue4, blue5, purple3])

   uv_cmap = matplotlib.colors.ListedColormap([purple5, purple4, purple3, \
             purple2, purple1, orange1, orange2, orange3, orange4, orange5])

   diff_cmap = matplotlib.colors.ListedColormap([blue7, blue6, blue5, blue4, \
               blue3, blue2, blue1, red1, red2, red3, red4, red5, red6, red7])

   mslp_cmap = matplotlib.colors.ListedColormap([purple7, purple6, purple5, \
               red7, red6, red5,orange7, orange6, orange5, green8, green7, \
               green6, blue6, blue5, blue4, gray3, gray2, gray1])

   paintball_colors = matplotlib.colors.ListedColormap([q1, b8, q3, q4, q5, q6, \
                      q7, q8, b6, q10, q11, b3, b2, purple5, red5, green5, \
                      blue5, orange5])

   paintball_colors_list = [q1, b8, q3, q4, q5, q6, q7, q8, b6, q10, q11, b3, \
                           b2, purple5, red5, green5, blue5, orange5]

   mslp_paint_colors = matplotlib.colors.ListedColormap([purple8,purple6, \
                       purple4, red8, red6, red4, orange8, orange6, orange4, \
                       green8, green6, green4, blue8, blue6, blue4, gray3, \
                       gray2, gray1])

   all_blues_cmap =  matplotlib.colors.ListedColormap([blue1, blue2, blue3, \
                     blue4, blue5, blue6, blue7, blue8, blue9])

   all_greens_cmap =  matplotlib.colors.ListedColormap([green1, green2, green3, \
                      green4, green5, green6, green7, green8, green9])

   all_reds_cmap =  matplotlib.colors.ListedColormap([red1, red2, red3, red4, \
                    red5, red6, red7, red8, red9])

   mfc_cmap = matplotlib.colors.ListedColormap([gray1, orange2, orange3, \
              orange4, red5, red6, red7])

   corf_cmap = matplotlib.colors.ListedColormap([purple7, purple5, purple3, \
               purple1, gray1, gray1, orange1, orange3, orange5, orange7])

   ul_dvg_cmap = matplotlib.colors.ListedColormap([green8, green6, green4, \
                 green2, gray1, gray1, purple2, purple4, purple6, purple8])

   rain_cmap = matplotlib.colors.ListedColormap([green2, green3, green5, \
               blue4, blue5, blue6, purple6, purple5, purple4, red4, red5, red6])

   cp_cmap   = matplotlib.colors.ListedColormap([gray8, gray6, purple6, \
               purple5, purple4, blue7, blue6, blue5, blue4, green6, green5, \
               green4, green3, orange6, orange5, orange4, red5, red6, red8])

   cwp_cmap = matplotlib.colors.ListedColormap([gray2, gray3, gray4, gray5, \
              gray6, blue6, blue5, blue4, blue3, green6, green5, green4, \
              green3, green2, orange2, orange3, orange4, orange5, red5, red6, \
              red7, red8, purple6, purple7, purple8])

   mslp_cmap = matplotlib.colors.ListedColormap([purple8, purple6, purple4, \
               red8, red6, red4, orange8, orange6, orange4, green8, green6, \
               green4, blue8, blue6, blue4, gray3, gray2, gray1])

   pw_cmap = matplotlib.colors.ListedColormap([orange8, orange7, orange6, orange5, orange4, orange3, \
             orange2, green1, green2, green3, green4, green5, green6, green7, \
             green8, green9, blue4, blue5, blue6, blue7, blue8, purple4, \
             purple5, purple6, purple7, red4, red5, red6, red7, red8])

   omega_cmap = matplotlib.colors.ListedColormap([purple7, purple5, purple3, \
                purple2, gray1, green2, green3, green5, green7])

   sw_cmap = matplotlib.colors.ListedColormap([blue9, blue8, blue7, blue6, blue5, \
             blue4, green8, green7, green6, green5, orange4, orange5, orange6, \
             orange7, red5, red6, red7, red8, purple7])

   fed_cmap = matplotlib.colors.ListedColormap([blue2, blue4, blue6, orange2, \
               orange4, orange6, red2, red4, red6, purple3, \
               purple5, purple7])
#######################################################################################


