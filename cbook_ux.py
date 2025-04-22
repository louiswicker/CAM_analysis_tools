import holoviews as holov
import cartopy.crs as ccrs
import uxarray as ux
import hvplot as hv
import geoviews.feature as gf
import metpy.plots as metplots

from matplotlib import colormaps


projection = projection = ccrs.PlateCarree()

Features = (
    gf.coastline(scale="110m", projection=ccrs.PlateCarree())
    * gf.borders(scale="110m", projection=ccrs.PlateCarree())
    * gf.states(scale="110m", projection=ccrs.PlateCarree())
)

def mpas_XYplot(uxds, pname=None, pvar=None, clim=(10,75), cmap='gist_ncar', Time=0, Level=10, features=Features, width=800, height=600, plot2=None):

    cmap = []
    for var in pvar:
        
        if var.upper()[0:3] == "REF":
            cmap.append(metplots.ctables.registry.get_colortable('NWSReflectivity'))
                                                           
        elif var.upper() == "CREF":
            cmap.append(metplots.ctables.registry.get_colortable('NWSReflectivity'))
            
        else:
            cmap.append(colormaps['viridis'])                                
            
    if plot2:
        theplot = holov.Layout(
            uxds[pvar[0]].isel(Time=Time, nVertLevels=Level).plot.polygons(rasterize=True,
                backend="bokeh",
                cmap=cmap[0],
                pixel_ratio=1.0,
                title=pname[0],
                clim=clim,
                fontscale=1.,
                width=800, height=600,
            ) * features.opts()
            +plot2[pvar[1]].isel(Time=Time, nVertLevels=Level).plot.polygons(rasterize=True,
                backend="bokeh",
                cmap=cmap[1],
                pixel_ratio=1.0,
                title=pname[1],
                clim=clim,
                fontscale=1.,
                width=800, height=600,
            ) * features.opts()
            ).cols(2)

    else:
        print("a single plot")
        theplot = holov.Layout(
                  uxds[pvar[0]].isel(Time=Time, nVertLevels=Level).plot.polygons(rasterize=True,
                backend="bokeh",
                cmap=cmap[0],
                pixel_ratio=1.0,
                title=pname[0],
                clim=clim,
                fontscale=1.,
                width=800, height=600,
            ) * features.opts())

    print("Done")

    return theplot
