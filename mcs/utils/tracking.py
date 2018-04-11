import numpy as np
import pickle
import pandas as pd
import os
from scipy.spatial.distance import pdist
import datetime
from geopandas import GeoDataFrame
from shapely.geometry import MultiPoint
from scipy.misc import imread

feature_list = ['area', 'convex_area', 'eccentricity', 
                'intense_area', 'convection_area',
                'convection_stratiform_ratio', 'intense_stratiform_ratio',
                'intense_convection_ratio', 'mean_intensity', 'max_intensity',
                'intensity_variance', 'major_axis_length', 'minor_axis_length',
                'solidity']

def to_datetime(time_string):

    if int(time_string[0:2]) > 90:
        year = "19" + time_string[0:2]
    else:
        year = "20" + time_string[0:2]

    month = time_string[2:4]
    day = time_string[4:6]
    hour = time_string[7:9]
    minute = time_string[9:11]

    return datetime.datetime(int(year), int(month), int(day), int(hour), int(minute))
    
    
def set_dates(df):

    dates = []

    for fn in df['filename']:

        dates.append(to_datetime(fn[-21:-10]))

    df.loc[:, 'datetime'] = dates
    
    return df
    
def get_geometry(df, slice_location):

    geometry = []

    for idx, row in df.iterrows():

        xmin = row['xmin']
        ymin = row['ymin']

        img = imread(slice_location + row['filename'], mode='P')

        y, x = np.where(img >= 10)

        polt = MultiPoint(np.array(list(zip(xmin+x, ymin+y)))).convex_hull

        geometry.append(polt)
        
    return geometry
    
def get_normalization(df):

    norm = []

    for col in feature_list:
        norm.append(np.max(df[col].values))
        
    return norm


def create_tracks(df, rng, prefix, crsr, ssr, p, slice_location, norm=None):

    #print("Selecting CRSR:", str(crsr), " SSR: ", str(ssr), " Probability: ", str(p))
    big_df = df[(df.CRSR==crsr) & (df.SSR==ssr) & (df.mcs_proba >= p)].copy()
    
    #print("Cacluating dates based on filenames")
    big_df = set_dates(big_df)

    #print("Cleaning up index")
    big_df = big_df.reset_index()
    
    #print("Calculating normalization factors for each feature")
    if norm is None:
        normalization = get_normalization(big_df)
    else:
        normalization = get_normalization(norm)

    #print("Calculating convex hull geometry")
    
    geo_df = GeoDataFrame(big_df, geometry=get_geometry(big_df, slice_location))

    #print("Cleaning up geodataframe index")
    geo_df = geo_df.reset_index(drop=True)

    #print("Initializing storm numbers")
    geo_df['storm_num'] = np.nan

    storm_num = 0

    #print("finding current times")
    cur_time = geo_df[geo_df['datetime'] == rng[0].to_pydatetime()]

    #print("setting initial storm numbers")
    #set storm numbers for the first time period
    for idx, row in cur_time.iterrows():

        geo_df.loc[idx, 'storm_num'] = storm_num

        storm_num += 1

    #print("running storm tracking")
    #run until the second to last time period
    for i in range(len(rng)-1):

        #print(crsr, ssr, p, rng[i])
            
        cur_time = geo_df[geo_df['datetime'] == rng[i].to_pydatetime()]

        next_time = geo_df[geo_df['datetime'] == rng[i+1].to_pydatetime()]

        if len(cur_time) > 0 and len(next_time) > 0:

            distance_matrix = np.ones(shape=(len(cur_time), len(next_time)), dtype=np.float) * np.nan

            for cc, (cid, crow) in enumerate(cur_time.iterrows()):
                for nc, (nid, nrow) in enumerate(next_time.iterrows()):

                    if crow['geometry'].intersects(nrow['geometry']):

                        distance_matrix[cc, nc] = pdist([crow[feature_list].values / normalization,
                                                         nrow[feature_list].values / normalization])

            a = np.copy(distance_matrix)

            while np.sum(~np.isnan(a)) > 0:

                track, candidate = np.where(a == np.nanmin(a))

                c_idx = next_time[candidate[0]:candidate[0]+1].index[0]
                t_idx = cur_time[track[0]:track[0]+1].index[0]

                next_time.loc[c_idx, 'storm_num'] = geo_df.loc[t_idx, 'storm_num']

                geo_df.loc[c_idx, 'storm_num'] = geo_df.loc[t_idx, 'storm_num']

                a[track[0], :] = np.nan

                a[:, candidate[0]] = np.nan

        new_storms = next_time[next_time['storm_num'].isnull()]

        for idx, row in new_storms.iterrows():

            geo_df.loc[idx, 'storm_num'] = storm_num

            storm_num += 1

    out_folder = "../data/track_data/unmatched/" + prefix
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        
    filename = out_folder + "/" + prefix + "_" + str(crsr).zfill(2) + "_" + str(ssr).zfill(3) + "_p" + str(int(p*100)).zfill(2) + ".pkl"
    pickle.dump(geo_df, open(filename, "wb"))
    
    print("Finished ", filename)
    
    
def rematch_tracks(df, rng, prefix, crsr, ssr, p, buffer_size=25, norm=None):

    df['storm_loc'] = 'm'

    grouped = df.groupby('storm_num')

    for gid, group in grouped:

        if len(group) >= 2:

            idx_s = group.index[0]
            idx_e = group.index[-1]

            df.loc[idx_s, 'storm_loc'] = 's'
            df.loc[idx_e, 'storm_loc'] = 'f'

    df['rematched'] = False
    
    if norm is None:
        normalization = get_normalization(df)
    else:
        normalization = get_normalization(norm)

    starts = []
    ends = []

    for d in rng:

        dfs = df[(pd.to_datetime(df.datetime)==d) & (df.storm_loc == 'f')]

        dff = df[(pd.to_datetime(df.datetime)>(d + datetime.timedelta(minutes=15))) & \
                 (pd.to_datetime(df.datetime)<=(d + datetime.timedelta(minutes=60))) & \
                 (df.storm_loc == 's')]

        if len(dfs) > 0 and len(dff) > 0:

            distance_matrix = np.ones(shape=(len(dfs), len(dff)), dtype=np.float) * np.nan

            for cc, (cid, crow) in enumerate(dfs.iterrows()):
                for nc, (nid, nrow) in enumerate(dff.iterrows()):

                    if crow['geometry'].buffer(buffer_size).intersects(nrow['geometry']):

                        distance_matrix[cc, nc] = pdist([crow[feature_list].values / normalization,
                                                         nrow[feature_list].values / normalization])

            a = np.copy(distance_matrix)

            while np.sum(~np.isnan(a)) > 0:

                track, candidate = np.where(a == np.nanmin(a))

                c_idx = dff[candidate[0]:candidate[0]+1].index[0]
                t_idx = dfs[track[0]:track[0]+1].index[0]

                cur_stormnum = dfs.loc[t_idx, 'storm_num']
                nex_stormnum = dff.loc[c_idx, 'storm_num']

                c_idx = df[df.storm_num==nex_stormnum].index.values

                df.loc[c_idx, 'storm_num'] = cur_stormnum
                df.loc[c_idx, 'rematched'] = True

                t_idx = df[df.storm_num==cur_stormnum].index.values

                df.loc[t_idx, 'rematched'] = True

                a[track[0], :] = np.nan

                a[:, candidate[0]] = np.nan
    
    out_folder = "../data/track_data/rematched/" + prefix
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        
    filename = out_folder + "/" + prefix + "_" + str(crsr).zfill(2) + "_" + str(ssr).zfill(3) + "_p" + str(int(p*100)).zfill(2) + ".pkl"
    pickle.dump(df, open(filename, "wb"))
    
    print("Finished", filename)