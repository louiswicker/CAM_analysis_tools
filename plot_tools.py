import numpy as np
import netCDF4 as ncdf
import matplotlib as mlab
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#---------------------------------------------------------------------------------------------------------------

def init_cartopy_plot(ncols=1, nrows=1, figsize=(10,10)):
    
    # Set up cartopy stuff here, so the plot routine is already set to use it.

    fig, axes = plt.subplots(figsize=figsize, ncols=ncols, nrows=nrows, 
        subplot_kw=dict(projection=ccrs.LambertConformal(central_latitude = 30, central_longitude = -95., standard_parallels = (20,20))))

    if nrows*ncols > 1:
        for ax in axes:
            ax.gridlines(draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
            ax.add_feature(cfeature.BORDERS, linestyle='solid', edgecolor='black', zorder=10)
            ax.add_feature(cfeature.STATES, linestyle='solid', edgecolor='black', zorder=10)
    else:
        axes.gridlines(draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
        axes.add_feature(cfeature.BORDERS, linestyle='solid', edgecolor='black', zorder=10)
        axes.add_feature(cfeature.STATES, linestyle='solid', edgecolor='black', zorder=10)

    return fig, axes

#---------------------------------------------------------------------------------------------------------------

def add_colorbar(plot, ax):
    """Add a vertical color bar to an image plot."""
    
    # cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    # plt.colorbar(plot, cax=cax, orientation='horizontal') # Similar to fig.colorbar(im, cax = cax)
    
    plt.colorbar(plot, ax=ax, shrink=0.75)

#---------------------------------------------------------------------------------------------------------------

def plot_w_from_xarray(ds, var='W', klevel=25, fhour=-1, title='', colormap='viridis', \
                       vmax=20, vmin=-10., contours=None, newlat=None, newlon=None, ax = None, cartopy=True, coords='latlonpres'):
     
    if ax != None:
        axes = ax
    else:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,12))
        cartopy = False

    ylim = None
    xlim = None

    clevels = np.linspace(vmin, vmax, 16)  
    
    if fhour < 0:
        fhour = int(len(ds.coords['fhour'])) - 1
               
    data = ds.W.isel(fhour=fhour, nz=klevel)
    
    if data.ndim > 2:
        data = data[0]

    if coords == 'latlonpres':
        lats = ds.lats
        lons = ds.lons
        pres = ds.pres
        if pres[0] > 1040.0:
            pres = pres/100.
                    
        if newlat == None:
            lat_min, lat_max = lats.min(), lats.max()
        else:
            lat_min, lat_max = newlat

        if newlon == None:
            lon_min, lon_max = lons.min(), lons.max()
        else:
            lon_min, lon_max = newlon
            
        print(f'\nPlot Lat Min: %4.1f  Lat Max:  %4.1f  ' % (lat_min, lat_max))
        print(f'\nPlot Lon Min: %4.1f  Lon Max:  %4.1f\n' % (lon_min, lon_max))
        
    frame_title = "FHOUR: %2.2i %s at Pres=%3.0f mb for %s \n Max: %3.1f   Min: %4.2f" % (fhour, var, pres[klevel], title, data.max(), data.min())
    
    if cartopy:
        plot = axes.pcolormesh(lons, lats, data, shading='auto', vmax=vmax, vmin=vmin, cmap=colormap, transform=ccrs.PlateCarree())
        
        if contours != None:
            axes.contour(lons, lats, data, levels=contours, colors=['blue', 'red'], linewidth=0.5, transform=ccrs.PlateCarree())
            
        axes.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
    else:
        plot = axes.pcolormesh(lons, lats, data, shading='auto', vmax=vmax, vmin=vmin, cmap=colormap)
        
        if contours != None:
            axes.contour(lons, lats, data, levels=contours, colors=['blue', 'red'], linewidth=0.5, )

        
    plt.colorbar(plot, ax=axes, shrink=0.75)
    
    axes.set_title(frame_title, fontsize=16)