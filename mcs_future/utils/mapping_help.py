
from math import floor
from matplotlib import patheffects
from matplotlib import patches as mpatches
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from netCDF4 import Dataset
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
from math import log
from math import exp
from copy import copy
import cartopy
import cartopy.crs as ccrs

import pickle
import numpy as np
import pandas as pd

from matplotlib import cm

def quantize_perc_agw(value):
    
    if value < 5:
        return 0
    elif value >= 5 and value < 15:
        return 1
    elif value >= 15 and value < 30:
        return 2
    elif value >= 30 and value < 60:
        return 3
    elif value >= 60 and value < 80:
        return 4
    elif value >= 80 and value < 120:
        return 5
    elif value >= 120 and value < 150:
        return 6
    elif value >= 150 and value < 200:
        return 7
    elif value >= 200 and value < 250:
        return 8
    elif value >= 250 and value < 300:
        return 9
    elif value >= 300:
        return 10
    elif np.isnan(value):
        return None
    
def draw_perc_agw(ax, geom, vals, debug=False):

    for g, val in zip(geom, vals):

        if np.isfinite(val):
            
            if debug:
                x = g.centroid.x
                y = g.centroid.y
            
                ax.text(x, y, str(int(val)), transform=ccrs.PlateCarree())

            quant = quantize_perc_agw(val)

            facecolor = cm.BrBG(quant/10)

        else:
            facecolor='grey'


        ax.add_geometries([g], ccrs.PlateCarree(),
                          facecolor=facecolor, edgecolor=facecolor)
                          
def quantize_perc_agw_prop_raw(value):
    
    val = np.zeros(shape=value.shape, dtype=int)
    
    y, x = np.where((value >= 10))
    val[y, x] = 1
    y, x = np.where((value >= 20))
    val[y, x] = 2
    y, x = np.where((value >= 30))
    val[y, x] = 3
    y, x = np.where((value >= 40))
    val[y, x] = 4
    y, x = np.where((value >= 50))
    val[y, x] = 5
    y, x = np.where((value >= 60))
    val[y, x] = 6
    y, x = np.where((value >= 70))
    val[y, x] = 7
    y, x = np.where((value >= 80))
    val[y, x] = 8
    y, x = np.where((value >= 90))
    val[y, x] = 9

    return val
    
def quantize_perc_agw_prop(value):
    
    if value < 10:
        return 0
    elif value >= 10 and value < 20:
        return 1
    elif value >= 20 and value < 30:
        return 2
    elif value >= 30 and value < 40:
        return 3
    elif value >= 40 and value < 50:
        return 4
    elif value >= 50 and value < 60:
        return 5
    elif value >= 60 and value < 70:
        return 6
    elif value >= 70 and value < 80:
        return 7
    elif value >= 80 and value < 90:
        return 8
    elif value >= 90:
        return 9
    elif np.isnan(value):
        return None
        
def draw_perc_agw_prop(ax, geom, vals, debug=False):

    for g, val in zip(geom, vals):

        if np.isfinite(val):
            
            if debug:
                x = g.centroid.x
                y = g.centroid.y
            
                ax.text(x, y, str(int(val)), transform=ccrs.PlateCarree())

            quant = quantize_perc_agw_prop(val)

            facecolor = cm.BrBG(quant/9)

        else:
            facecolor='grey'


        ax.add_geometries([g], ccrs.PlateCarree(),
                          facecolor=facecolor, edgecolor=facecolor)

def wrf_to_lon_lat(lons, lats, x, y):
    
    longs = [lons[x1, y1] for (x1,y1) in zip(y, x)]
    latis = [lats[x1, y1] for (x1,y1) in zip(y, x)]
    
    return longs, latis


def get_point_subset(df, outline, wrf_ref=None, within=True):
    
    xc = np.array([np.mean([x, y]) for (x, y) in zip(df.xmin.values, df.xmax.values)])
    yc = np.array([np.mean([x, y]) for (x, y) in zip(df.ymin.values, df.ymax.values)])

    if wrf_ref is not None:

        nc = Dataset(wrf_ref)

        lons = nc.variables['XLONG'][:,:]
        lats = nc.variables['XLAT'][:,:]

        lo, la = wrf_to_lon_lat(lons, lats, xc.astype(int), yc.astype(int))

    else:

        lo, la = NOWrad_to_lon_lat(np.array(xc), np.array(yc))

    df['lats'] = la
    df['lons'] = lo

    geometry = [Point(xy) for xy in zip(df.lons, df.lats)]
    df = df.drop(['lons', 'lats'], axis=1)
    crs = {'init': 'epsg:4326'}
    points = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)

    points_ = gpd.sjoin(points, outline, how="inner", op='within')
    
    return points_
    
def draw_states(ax):
        
    shapename = 'admin_1_states_provinces_lakes_shp'
    states_shp = shpreader.natural_earth(resolution='50m',
                                     category='cultural', name=shapename)

    for state, info in zip(shpreader.Reader(states_shp).geometries(), shpreader.Reader(states_shp).records()):
        if info.attributes['admin'] == 'United States of America':

            ax.add_geometries([state], ccrs.PlateCarree(),
                              facecolor='None', edgecolor='k')
            
def find_side(ls, side):
    """From http://nbviewer.jupyter.org/gist/ajdawson/dd536f786741e987ae4e
    Given a shapely LineString which is assumed to be rectangular, return the
    line corresponding to a given side of the rectangle."""
    
    minx, miny, maxx, maxy = ls.bounds
    points = {'left': [(minx, miny), (minx, maxy)],
              'right': [(maxx, miny), (maxx, maxy)],
              'bottom': [(minx, miny), (maxx, miny)],
              'top': [(minx, maxy), (maxx, maxy)],}
    return sgeom.LineString(points[side])


def lambert_xticks(ax, ticks, fontsize=12):
    """From http://nbviewer.jupyter.org/gist/ajdawson/dd536f786741e987ae4e
    Draw ticks on the bottom x-axis of a Lambert Conformal projection."""
    
    te = lambda xy: xy[0]
    lc = lambda t, n, b: np.vstack((np.zeros(n) + t, np.linspace(b[2], b[3], n))).T
    xticks, xticklabels = _lambert_ticks(ax, ticks, 'bottom', lc, te)
    ax.xaxis.tick_bottom()
    ax.set_xticks(xticks)
    ax.set_xticklabels([ax.xaxis.get_major_formatter()(xtick) for xtick in xticklabels], fontsize=fontsize)
    

def lambert_yticks(ax, ticks, fontsize=12):
    """From http://nbviewer.jupyter.org/gist/ajdawson/dd536f786741e987ae4e
    Draw ricks on the left y-axis of a Lamber Conformal projection."""
    
    te = lambda xy: xy[1]
    lc = lambda t, n, b: np.vstack((np.linspace(b[0], b[1], n), np.zeros(n) + t)).T
    yticks, yticklabels = _lambert_ticks(ax, ticks, 'left', lc, te)
    ax.yaxis.tick_left()
    ax.set_yticks(yticks)
    ax.set_yticklabels([ax.yaxis.get_major_formatter()(ytick) for ytick in yticklabels], fontsize=fontsize)

def _lambert_ticks(ax, ticks, tick_location, line_constructor, tick_extractor):
    """From http://nbviewer.jupyter.org/gist/ajdawson/dd536f786741e987ae4e
    Get the tick locations and labels for an axis of a Lambert Conformal projection."""
    
    outline_patch = sgeom.LineString(ax.outline_patch.get_path().vertices.tolist())
    axis = find_side(outline_patch, tick_location)
    n_steps = 30
    extent = ax.get_extent(ccrs.PlateCarree())
    _ticks = []
    for t in ticks:
        xy = line_constructor(t, n_steps, extent)
        proj_xyz = ax.projection.transform_points(ccrs.Geodetic(), xy[:, 0], xy[:, 1])
        xyt = proj_xyz[..., :2]
        ls = sgeom.LineString(xyt.tolist())
        locs = axis.intersection(ls)
        if not locs:
            tick = [None]
        else:
            tick = tick_extractor(locs.xy)
        _ticks.append(tick[0])
    # Remove ticks that aren't visible:    
    ticklabels = copy(ticks)
    while True:
        try:
            index = _ticks.index(None)
        except ValueError:
            break
        _ticks.pop(index)
        ticklabels.pop(index)
    return _ticks, ticklabels
    
def quantize_season_raw(value):
    
    val = np.zeros(shape=value.shape, dtype=int)
    
    y, x = np.where((value >= 5))
    val[y, x] = 1
    y, x = np.where((value >= 10))
    val[y, x] = 2
    y, x = np.where((value >= 20))
    val[y, x] = 3
    y, x = np.where((value >= 30))
    val[y, x] = 4
    y, x = np.where((value >= 60))
    val[y, x] = 5
    y, x = np.where((value >= 90))
    val[y, x] = 6
    y, x = np.where((value >= 120))
    val[y, x] = 7
    y, x = np.where((value >= 150))
    val[y, x] = 8

    return val
    
def quantize_season(value):
    
    if value > 0 and value < 5:
        return 0
    elif value >= 5 and value < 10:
        return 1
    elif value >= 10 and value < 20:
        return 2
    elif value >= 20 and value < 30:
        return 3
    elif value >= 30 and value < 60:
        return 4
    elif value >= 60 and value < 90:
        return 5
    elif value >= 90 and value < 120:
        return 6
    elif value >= 120:
        return 7

def draw_states(ax):
        
    shapename = 'admin_1_states_provinces_lakes_shp'
    states_shp = shpreader.natural_earth(resolution='50m',
                                     category='cultural', name=shapename)

    for state, info in zip(shpreader.Reader(states_shp).geometries(), shpreader.Reader(states_shp).records()):
        if info.attributes['admin'] == 'United States of America':

            ax.add_geometries([state], ccrs.PlateCarree(),
                              facecolor='None', edgecolor='k')
                              
def draw_midwest(ax):
    
    shapename = "../data/shapefiles/map/midwest_outline_latlon_grids"
    shp = shpreader.Reader(shapename)
    for outline, info in zip(shp.geometries(), shp.records()):
        ax.add_geometries([outline], ccrs.PlateCarree(),
                          facecolor='None', edgecolor='k', linewidth=4)
                              
def get_season_mcs(run, season, dbz, mw=False):
    
    shapename = "../data/shapefiles/raw_data/shapefiles_day/" + run + "/" + season + '_' + dbz + '_pgw'
    shp = shpreader.Reader(shapename)
    geom = shp.geometries()
    
    if mw == False:
        mcs_vals = np.array([a.attributes['count'] for a in shp.records()])
        mcs_vals[~np.isfinite(mcs_vals)] = 0
        return geom, mcs_vals
    
    else:
        mcs_vals = []
        for a in shp.records():
            if a.attributes['midwest'] == True:
                mcs_vals.append(a.attributes['count'])
        mcs_vals = np.array(mcs_vals)
        return mcs_vals
        
def generate_view(plt, w_lon, e_lon, n_lat, s_lat, from_proj, to_proj):

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
    
def draw_grids_season(ax, geom, vals, classes, debug=False):

    for g, val in zip(geom, vals):

        if np.isfinite(val) and val > 0:
            
            if debug:
                x = g.centroid.x
                y = g.centroid.y
            
                ax.text(x, y, str(int(val)), transform=ccrs.PlateCarree())

            quant = quantize_season(val)

            facecolor = cm.viridis(quant/classes)

        else:
            facecolor='grey'


        ax.add_geometries([g], ccrs.PlateCarree(),
                          facecolor=facecolor, edgecolor=facecolor)