from preprocess import *
import pandas as pd
import numpy as np

def crps(y_true, y_pred):
    ans = 0
    ground_t = y_true.argmax(1)
    for i, t in enumerate(ground_t):
        for n in range(-99, 100):
            h = n>=(t-99)
            
            ans+=(y_pred[i][n+99]-h)**2
            
    return ans/(199*len(y_true))

# Augment function
# playIDを reindexする!!!
def flip_y_afterFE(df):
    df['Y'] = -df['Y']
    df['Orientation'] = 180 - df['Orientation']
    df.loc[df['Orientation']<0, 'Orientation'] = 360 + df['Orientation']
    df['Dir'] = 180 - df['Dir']
    df.loc[df['Dir']<0, 'Dir'] = 360 + df['Dir']
    df = angle_rad(df)
    df['v_speed'] = -df['v_speed']
    # miss
    df['spped_vert_rusher'] = -df['spped_vert_rusher']
    df['sita_diff_from_runner'] = -df['sita_diff_from_runner']
    df['0.5later_v_speed'] = -df['0.5later_v_speed']
    return df

def augment(df):
    dfs = [df]
    df_aug = flip_y_afterFE(df.copy())
    dfs.append(df_aug)
    df = pd.concat(dfs, axis=0)
    return df


def split_y(df, k=5):
    ys = df['Yards'][::22].values
    ret = np.zeros((ys.shape[0], k))
    sorted_ys = sorted(ys)
    bins = []
    for i in range(k):
        bins.append(sorted_ys[int(ys.shape[0]*(i/k))])
    bins.append(sorted_ys[-1])
    for i in range(ys.shape[0]):
        for j in range(k):
            if ys[i]>=bins[j] and ys[i]<=bins[j+1]:
                ret[i,j] = 1
                break
    return ret