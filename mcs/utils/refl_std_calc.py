import sys, getopt

from scipy.ndimage import imread
import numpy as np
import pickle
import pandas as pd

from multiprocessing import Pool

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

feature_list = ['area', 'convex_area', 'eccentricity', 
                'intense_area', 'convection_area',
                'convection_stratiform_ratio', 'intense_stratiform_ratio',
                'intense_convection_ratio', 'mean_intensity', 'max_intensity',
                'intensity_variance', 'major_axis_length', 'minor_axis_length',
                'solidity']

def get_mean_dur(crsr, ssr, p, pref, year):

    entry = {'crsr':[], 'ssr':[], 'p':[], 'mean_size':[], 'dist':[]}
    entry['crsr'].append(crsr)
    entry['ssr'].append(ssr)
    entry['p'].append(float(p))

    fn = "../data/track_data/" + "/" + pref + "/" + str(year) + "/" + str(year) + "_" + str(crsr).zfill(2) + "_" + str(ssr).zfill(3) + "_p" + str(int(p*100)).zfill(2) + ".pkl"


    bg = pickle.load(open(fn, 'rb'))
    df1 = bg[(bg.CRSR==crsr) & (bg.SSR==ssr)]
    
    grouped = df1.groupby('storm_num')

    tdata = []
    size = []
    for gid, group in grouped:
        duration = (pd.to_datetime(group.iloc[-1]['datetime']) - pd.to_datetime(group.iloc[0]['datetime'])).total_seconds() / 3600
        if duration >= 0.5:
            
            tdata.append(duration)

    entry['dist'].append(np.array(tdata))
    
    print("CRSR: ", crsr, "SSR:", ssr, "MCS_P:", p, "Mean length:", np.mean(tdata))
    
    return entry
    
    
def get_lin_err(crsr, ssr, p, pref, year):

    entry = {'crsr':[], 'ssr':[], 'p':[], 'mean_size':[], 'dist':[]}
    entry['crsr'].append(crsr)
    entry['ssr'].append(ssr)
    entry['p'].append(float(p))

    fn = "../data/track_data/" + "/" + pref + "/" + str(year) + "/" + str(year) + "_" + str(crsr).zfill(2) + "_" + str(ssr).zfill(3) + "_p" + str(int(p*100)).zfill(2) + ".pkl"


    bg = pickle.load(open(fn, 'rb'))
    df1 = bg[(bg.CRSR==crsr) & (bg.SSR==ssr)]

    grouped = df1.groupby('storm_num')

    tdata = []
    for gid, group in grouped:
        duration = (pd.to_datetime(group.iloc[-1]['datetime']) - pd.to_datetime(group.iloc[0]['datetime'])).total_seconds() / 3600
        if duration >= 1:

            xc = [np.mean([x, y])*2 for (x, y) in zip(group['xmin'].values, group['xmax'].values)]
            yc = [np.mean([x, y])*2 for (x, y) in zip(group['ymin'].values, group['ymax'].values)]

            xcmin = np.min(xc)
            xcmax = np.max(xc)

            ycmin = np.min(yc)
            ycmax = np.max(xc)

            x = [[x1] for x1 in xc]
            
            clf = LinearRegression()

            clf.fit(x, np.array(yc))
            
            y = clf.predict(x)
            
            rmse = np.sqrt(mean_squared_error(yc, y))
            
            tdata.append(rmse)

    entry['dist'].append(np.array(tdata))

    print("CRSR: ", crsr, "SSR:", ssr, "MCS_P:", p, "Mean Linearity Error:", np.mean(tdata))
    
    return entry
    
            
def get_std_refl(crsr, ssr, p, pref, year):
    
    entry = {'crsr':[], 'ssr':[], 'p':[], 'mean_size':[], 'dist':[]}
    entry['crsr'].append(crsr)
    entry['ssr'].append(ssr)
    entry['p'].append(float(p))
    
    fn = "../data/track_data/" + pref + "/" + str(year) + "/" + str(year) + "_" + str(crsr).zfill(2) + "_" + str(ssr).zfill(3) + "_p" + str(int(p*100)).zfill(2) + ".pkl"

    #fn = "2015/" + str(crsr).zfill(2) + "_" + str(ssr).zfill(2) + "_p" + str(int(p*100)) + "_" + pref + "_tracks.pkl"

    print(crsr, ssr, p, pref, year, fn)
    bg = pickle.load(open(fn, 'rb'))
    df1 = bg[(bg.CRSR==crsr) & (bg.SSR==ssr)]
    
    gb = []

    for col in feature_list:
        gb.append(np.max(df1[col].values))
    
    grouped = df1.groupby('storm_num')
    
    
    tdata = []
    size = []
    for gid, group in grouped:
        duration = (pd.to_datetime(group.iloc[-1]['datetime']) - pd.to_datetime(group.iloc[0]['datetime'])).total_seconds() / 3600
        if duration >= 1:
            
            xmin = np.min(group['xmin'])
            xmax = np.max(group['xmax'])
            ymin = np.min(group['ymin'])
            ymax = np.max(group['ymax'])
            
            res = np.zeros(shape=(len(group), 1+ymax-ymin, 1+xmax-xmin), dtype=np.uint8)
            
            for idx, (rid, row) in enumerate(group.iterrows()):

                img = imread(row['filename'], mode='P')

                y, x = np.where(img>0)

                res[idx, y, x] = 5*img[y, x]

            a = res.flatten()
            tdata.append(np.std(a[a>0]))
    
    entry['dist'].append(np.array(tdata, dtype=float))
    
    print("CRSR: ", crsr, "SSR:", ssr, "MCS_P:", p, "Mean std:", np.mean(tdata))
    
    return entry
    
    
            
if __name__ == "__main__":

    metric = None
    
    argv = sys.argv[1:]
    
    try:                                
        opts, args = getopt.getopt(argv, "hm:n", ["metric="])
    except getopt.GetoptError as e: 
        print(e)
        sys.exit(2)                     
    for opt, arg in opts: 
        print("arg:", arg, "opt:", opt)     
        if opt in ("-m", "--metric"): 
       
            metric = arg                  
    
    print(metric)
    
    crsr_ = [6, 6, 6, 12, 12, 12, 24, 24, 24, 48, 48, 48]
    ssr_ = [48, 96, 192, 48, 96, 192, 48, 96, 192, 48, 96, 192]
    
    entries = []
    
    pref = "rematched"
    year = 2016
    
    for p in [0.0, 0.5, 0.9, 0.95]:

        pobj = Pool(12)
        
        if metric == 'std_refl':
            result = [pobj.apply_async(get_std_refl, (crsr, ssr, p, pref, year)) for (crsr, ssr) in zip(crsr_, ssr_)]
            
        elif metric == 'lin_err':
            result = [pobj.apply_async(get_lin_err, (crsr, ssr, p, pref, year)) for (crsr, ssr) in zip(crsr_, ssr_)]
            
        elif metric == 'mean_dur':
            result = [pobj.apply_async(get_mean_dur, (crsr, ssr, p, pref, year)) for (crsr, ssr) in zip(crsr_, ssr_)]
            
        else:
            print("metric isn't available")
            sys.exit(2) 
            break

        pobj.close()
        pobj.join()
        
        for i in result:
            entry = i.get()
            
            df = pd.DataFrame(columns=['CRSR', 'SSR', 'MCS_proba', 'Distribution'])
            
            df['CRSR'] = entry['crsr']
            df['SSR'] = entry['ssr']
            df['MCS_proba'] = entry['p']
            df['Distribution'] = entry['dist']
            
            df['mean'] = [np.mean(x) for x in df['Distribution'].values]
            df['median'] = [np.median(x) for x in df['Distribution'].values]
            df['sd'] = [np.std(x) for x in df['Distribution'].values]
            
            entries.append(df)
    
    df = pd.concat(entries)

    pickle.dump(df, open(str(year) + "_" + metric + "_" + pref + "_master.pkl", "wb"))
    
    
    