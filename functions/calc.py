import pandas as pd
import math
import numpy as np

def distance(lines):
    length = []
    for line in lines:
        x1 = line[0] 
        y1 = line[1] 
        x2 = line[2] 
        y2 = line[3]
        l = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        length.append(l)
    return length

def calc_slope(lines):
    df_cols = ['x1', 'y1', 'x2', 'y2']

    df = pd.DataFrame(data=lines, columns= df_cols)

    df['line_length'] = distance(lines)
    df['slope'] = (df['y1']- df['y2']) / (df['x1'] - df['x2'])
    df['c'] = df['y1'] - df['slope'] * df['x1']

    return df 

def filter_lines(df, bin=30):
    straight_df = df[(df['slope']==np.inf)| (df['slope']== -np.inf) | (df['slope']== 0)]
    inclined_df = df[(df['slope']!=np.inf)& (df['slope']!= -np.inf) & (df['slope']!= 0)]

    # Create a linspace to group lines based on their closeness in slope and contant value, the groups are determined by the bin value 
    slope_bins = np.linspace((inclined_df['slope'].min() -0.001), (inclined_df['slope'].max()+0.001), num=bin)
    c_bins = np.linspace((inclined_df['c'].min()-0.001), (inclined_df['c'].max()+0.001), num=bin)
    
    inclined_df['slope_bins'] = pd.cut(inclined_df['slope'], slope_bins)
    inclined_df['c_bins'] = pd.cut(inclined_df['c'], c_bins)

    # group lines based on the clossness and calculate the average line slopes and c
    df_g = inclined_df.groupby(['c_bins', 'slope_bins'], as_index=False).mean()

    df_G = df_g.dropna()

    df_G['x1'] = df_G['x1'].astype(int)
    df_G['y1'] = df_G['y1'].astype(int)
    df_G['x2'] = df_G['x2'].astype(int)
    df_G['y2'] = df_G['y2'].astype(int)

    return straight_df, inclined_df, df_G 
