import matplotlib as mpl


def radar_colormap():
    nws_reflectivity_colors = [
    "#646464", # 0
    "#04e9e7", # 5
    "#019ff4", # 10
    "#0300f4", # 15
    "#02fd02", # 20
    "#01c501", # 25
    "#008e00", # 30
    "#fdf802", # 35
    "#e5bc00", # 40
    "#fd9500", # 45
    "#fd0000", # 50
    "#d40000", # 55
    "#bc0000", # 60
    "#f800fd", # 65
    "#9854c6", # 70
    "#fdfdfd", # 75
    "#000000"
    ]

    cmap = mpl.colors.ListedColormap(nws_reflectivity_colors)

    return cmap

def quantize(img, s=4, c=8, i=10):
    
    strat = 1*(img >= s)
    conv = 1*(img >= c)
    ints = 1*(img >= i)
    
    return strat+conv+ints