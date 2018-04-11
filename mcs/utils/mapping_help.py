
from math import floor
from matplotlib import patheffects
from matplotlib import patches as mpatches
import cartopy.io.shapereader as shpreader
import cartopy
import cartopy.crs as ccrs

import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def get_track_centroids(group):

    xc = [np.mean([x, y]) for (x, y) in zip(group['xmin'].values, group['xmax'].values)]
    yc = [np.mean([x, y]) for (x, y) in zip(group['ymin'].values, group['ymax'].values)]
    
    return xc, yc

def running_ave(arr, N):
    
    data = []
    length = len(arr)
    
    n = int(N/2)
    
    for i in range(length):
        
        if i + n + 1 > length:
            data.append(np.mean(arr[i-n:]))
            
        if i - n < 0:
            data.append(np.mean(arr[:i+n+1]))
        
        else:
            data.append(np.mean(arr[i-n:i+n+1]))
            
    return np.array(data)

def quantize(img):
    
    strat = 1*(img>=4)
    conv = 1*(img>=8)
    ints = 1*(img>=10)
    
    return strat+conv+ints

def utm_from_lon(lon):
    """
    utm_from_lon - UTM zone for a longitude

    Not right for some polar regions (Norway, Svalbard, Antartica)

    :param float lon: longitude
    :return: UTM zone number
    :rtype: int
    """
    return floor( ( lon + 180 ) / 6) + 1

def scale_bar(ax, proj, length, location=(0.5, 0.05), linewidth=3,
              units='km', m_per_unit=1000, fontsize=8):
    """

    http://stackoverflow.com/a/35705477/1072212
    ax is the axes to draw the scalebar on.
    proj is the projection the axes are in
    location is center of the scalebar in axis coordinates ie. 0.5 is the middle of the plot
    length is the length of the scalebar in km.
    linewidth is the thickness of the scalebar.
    units is the name of the unit
    m_per_unit is the number of meters in a unit
    """
    # find lat/lon center to find best UTM zone
    x0, x1, y0, y1 = ax.get_extent(proj.as_geodetic())
    # Projection in metres
    utm = ccrs.UTM(utm_from_lon((x0+x1)/2))
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(utm)
    # Turn the specified scalebar location into coordinates in metres
    sbcx, sbcy = x0 + (x1 - x0) * location[0], y0 + (y1 - y0) * location[1]
    # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbcx - length * m_per_unit/2, sbcx + length * m_per_unit/2]
    # buffer for scalebar
    buffer = [patheffects.withStroke(linewidth=5, foreground="w")]
    # Plot the scalebar with buffer
    ax.plot(bar_xs, [sbcy, sbcy], transform=utm, color='k',
        linewidth=linewidth, path_effects=buffer)
    # buffer for text
    buffer = [patheffects.withStroke(linewidth=3, foreground="w")]
    # Plot the scalebar label
    t0 = ax.text(sbcx, sbcy+10000, str(length) + ' ' + units, transform=utm,
        horizontalalignment='center', verticalalignment='bottom',
        path_effects=buffer, zorder=2, fontsize=fontsize)
    left = x0+(x1-x0)*0.05
    # Plot the N arrow
    t1 = ax.text(left, sbcy, u'\u25B2\nN', transform=utm,
        horizontalalignment='center', fontsize=20, verticalalignment='bottom',
        path_effects=buffer, zorder=2)
    # Plot the scalebar without buffer, in case covered by text buffer
    ax.plot(bar_xs, [sbcy, sbcy], transform=utm, color='k',
        linewidth=linewidth, zorder=3)
    
def quantize(img):
    
    strat = 1*(img>=4)
    conv = 1*(img>=8)
    ints = 1*(img>=10)
    
    return strat+conv+ints

def running_ave(arr, N):
    
    data = []
    length = len(arr)
    
    n = int(N/2)
    
    for i in range(length):
        
        if i + n + 1 > length:
            data.append(np.mean(arr[i-n:]))
            
        if i - n < 0:
            data.append(np.mean(arr[:i+n+1]))
        
        else:
            data.append(np.mean(arr[i-n:i+n+1]))
            
    return np.array(data)


def generate_view(w_lon, e_lon, n_lat, s_lat, from_proj, to_proj):

    view = plt.axes([0,0,1,1], projection=to_proj)

    view.set_extent([w_lon, e_lon, s_lat, n_lat])

    shapename = 'admin_1_states_provinces_lakes_shp'
    states_shp = shpreader.natural_earth(resolution='50m',
                                         category='cultural', name=shapename)

    for state, info in zip(shpreader.Reader(states_shp).geometries(), shpreader.Reader(states_shp).records()):
        if info.attributes['admin'] == 'United States of America':

            view.add_geometries([state], ccrs.PlateCarree(),
                              facecolor='None', edgecolor='k')
            
    return view

    
def NOWrad_to_lon_lat(xpoints, ypoints, xMin=0, yMin=0):
    """Convert NOWrad x,y grid coordinates to latitude,
    longitude coordinates.
    
    Can even convert a subset of image if you know 
    the west and north edges.
    
    Parameters
    ----------
    xpoints: (N, ) ndarray
        Array of x coordinates to be converted
    ypoints: (N, ) ndarray
        Array of y coordinates to be converted
    xMin: int
        Relative most westward x coordinate if image is clipped
    yMin: int
        Relative most northward y coordinate if image is clipped
    """

    #See: NOWrad Technical Note
    lats = 53 - (yMin + ypoints) * .0180
    lons = (xMin + xpoints) * .0191 - 130
    
    return lons, lats
    
def get_NOWrad_conus_lon_lat():

    x = np.array(list(range(0,3661)))
    y = np.array(list(range(0,1837)))

    return NOWrad_to_lon_lat(x, y)