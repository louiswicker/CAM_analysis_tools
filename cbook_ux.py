import holoviews as holov
import cartopy.crs as ccrs
import uxarray as ux
import hvplot as hv
import geoviews.feature as gf
import metpy.plots as metplots
import pandas as pd

from matplotlib import colormaps

from timeit import default_timer as timer

projection = projection = ccrs.PlateCarree()

_features = (
    gf.coastline(scale="110m", projection=ccrs.PlateCarree())
    * gf.borders(scale="110m", projection=ccrs.PlateCarree())
    * gf.states(scale="50m", projection=ccrs.PlateCarree())
)

_image_size  = [800,700]
_backend     = "bokeh"
_pixel_ratio = 1.0
_rasterize   = True
_ncols       = 2
_nrows       = 1

def mpas_XYplot(*args, **kwargs):

    backend      = kwargs.get("backend",     _backend)
    pixel_ratio  = kwargs.get("pixel_ratio", _pixel_ratio)
    rasterize    = kwargs.get("rasterize",   _rasterize)
    image_size   = kwargs.get("image_size",  _image_size)
    features     = kwargs.get("features",    _features)
    ncols        = kwargs.get("ncols",       _ncols)
    nrows        = kwargs.get("nrows",       _nrows)

    start = timer()
    cmap = []

    width  = image_size[0]
    height = image_size[1]

    if len(args) == 2: 

        print(f"MPAS_XY PLot:  2 UXDS data sets to be plotted")
        
        p1uxds  = args[0][0]
        p1label = args[0][2]
        p1attr  = args[0][1][p1label]
        p2uxds  = args[1][0]
        p2label = args[1][2]
        p2attr  = args[1][1][p2label]
        
    else:

        print(f"MPAS_XY PLot:  1 UXDS data sets to be plotted")

        p1uxds  = args[0][0]
        p1label = args[0][2]
        p1attr  = args[0][1][p1label]

    if len(args) == 2: 
        
        theplot = holov.Layout(
            p1uxds[p1attr[0]].isel(Time=p1attr[1], nVertLevels=p1attr[2]).plot.polygons(rasterize=_rasterize,
                backend=backend,
                cmap=p1attr[4],
                pixel_ratio=pixel_ratio,
                title=f"{p1label}:     Level = {p1attr[2]:02}       Time: {pd.to_datetime(p1uxds.Time[p1attr[1]].values)}",
                clim=p1attr[3],
                fontscale=1.,
                width=width, 
                height=height,
            ) * features.opts()
            +p2uxds[p2attr[0]].isel(Time=p2attr[1], nVertLevels=p2attr[2]).plot.polygons(rasterize=_rasterize,
                backend=backend,
                cmap=p2attr[4],
                pixel_ratio=pixel_ratio,
                title=f"{p2label}:     Level = {p2attr[2]:02}       Time: {pd.to_datetime(p1uxds.Time[p2attr[1]].values)}",
                clim=p2attr[3],
                fontscale=1.,
                width=width, 
                height=height,
            ) * features.opts()
            ).cols(ncols)

    else:
        theplot = p1uxds[p1attr[0]].isel(Time=p1attr[1], nVertLevels=p1attr[2]).plot.polygons(rasterize=_rasterize,
                backend=backend,
                cmap=p1attr[4],
                pixel_ratio=pixel_ratio,
                title=f"{p1label}:     Level = {p1attr[2]:02}       Time: {pd.to_datetime(p1uxds.Time[p1attr[1]].values)}",
                clim=p1attr[3],
                fontscale=1.,
                width=width, 
                height=height,
            ) * features.opts()

    end = timer()
    
    print(f"MPAS_XY PLot:  Time to create plot: {end - start:.2f} sec")

    return theplot
