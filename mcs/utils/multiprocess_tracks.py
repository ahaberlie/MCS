import sys, getopt

from tracking import rematch_tracks, create_tracks

from scipy.ndimage import imread
import numpy as np
import pickle
import pandas as pd
from scipy.spatial.distance import pdist
import datetime
from functools import partial


from multiprocessing import Pool

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def get_date_range(argv):

    start_date = ''
    end_date = ''
   
    try:
        opts, args = getopt.getopt(argv, "hs:e:", ["sdate=","edate="])
    except getopt.GetoptError:
        print('make_slices.py -s <start_date> -e <end_date>')
        sys.exit(2)
        
    for opt, args in opts:
        
        if opt == '-s':
            
            start_date = args
            
        elif opt == '-e':
        
            end_date = args
          
    print("Starting date:", start_date)
    print("Ending date:", end_date)
    
    return start_date, end_date

def creation_of_tracks(p, df, dt):
        
    years = df.groupby(df.index.year)
    for year, year_rows in years:
    
        print(year)
                
        rng = pd.date_range(datetime.datetime(year, 5, 1, 0, 0), datetime.datetime(year, 10, 1, 0, 0), freq=dt)
        
        year_rows = year_rows[year_rows.index.isin(rng.values)]
        
        year_rows.loc[:, 'datetime'] = year_rows.index.values
        
        year_rows.reset_index(drop=True, inplace=True)
                
        matching = []
    
        pobj = Pool(6)
        print("initializing async..")
        for crsr in [6, 12, 24, 48]:
            for ssr in [48, 96, 192]:

                matching.append(pobj.apply_async(create_tracks, (year_rows.copy(), rng, str(year), crsr, ssr, p, "")))
        
        pobj.close()
        pobj.join()
        
        for i in matching:
            res = i.get()
        
def rematching_tracks(p, dt):
   
    print("reading year files...")

    for year in range(2016, 2017):
            
        pobj = Pool(6)
        results = []
        
        for crsr in [6, 12, 24, 48]:
            for ssr in [48, 96, 192]:

                fname = "../data/track_data/unmatched/" + str(year) + "/" + str(year) + "_" + str(crsr).zfill(2) + "_" + str(ssr).zfill(3) + "_p" + str(int(p*100)).zfill(2) + ".pkl"
                print("reading file:", fname)
                
                try:
                    df = pickle.load(open(fname, "rb"))
                    
                    df.loc[:, 'datetime'] = pd.to_datetime(df.datetime)

                    df = df.set_index('datetime')
                    
                    rng = pd.date_range(datetime.datetime(year, 5, 1, 0, 0), datetime.datetime(year, 10, 1, 0, 0), freq=dt)
                    
                    df.loc[:, 'datetime'] = df.index.values
                
                    df.reset_index(drop=True, inplace=True)
                    
                    results.append(pobj.apply_async(rematch_tracks, (df.copy(), rng, str(year), crsr, ssr, p)))
                    
                except Exception as e:
                    print(e)
    
        pobj.close()
        pobj.join()
        
        for i in results:
            print(i.get())
        
        
    
if __name__ == "__main__":

    print("reading index file...")
    
    start_date, end_date = get_date_range(sys.argv[1:])
    
    df = pd.read_csv("../data/slice_data/labeled_slices_020618.csv")

    df.apply(partial(pd.to_numeric, errors='ignore'))
    
    df.info()
    
    df.loc[:, 'datetime'] = pd.to_datetime(df.datetime)

    df = df.set_index('datetime')
    
    df = df[start_date:end_date]
    
    df = df.drop('Unnamed: 0', axis=1)
    df = df.drop('Unnamed: 0.1', axis=1)
    df = df.drop('Unnamed: 0.1.1', axis=1)

    print("finished reading index file..")
    
    print("Calculating attributes..")
    fname = []

    for rid, row in df.iterrows():

        if rid.year == 2015:
        
            fn = "E:/p12_slices/92017_slices/2015/" + row.filename
        
        if rid.year == 2016:
        
            fn = "F:/2016/" + row.filename
            
        fname.append(fn)

    df.loc[:, 'filename'] = fname

    print("finished calculating attributes..")
    

    for p in [0.0, 0.5, 0.90, 0.95]:
        creation_of_tracks(p, df, '15T')
        rematching_tracks(p, '15T')
    