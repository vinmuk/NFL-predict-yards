import math
import datetime
import pandas as pd
import numpy as np

# feature engineering parameters
SORT_ORDER = ['PlayId', 'is_offence', 'is_rusher', 'rdist', 'JerseyNumber']

NEW_R_TIME = [0.5,1,1.5]

# t秒後の x,y std
LATER_STD_T = [1]
# 周辺のプレイヤー数
NEAR_YARD = [0.5, 1, 3, 5] # 近隣と定義するヤード数のリスト (二乗ではない)
NEAR_USE_ENE = True
NEAR_USE_FRI = False

# t秒後の周辺のプレイヤー数
LATER_NEAR_NEAR = [0.5,1,3,5]
LATER_NEAR_T = [0.5,1,1.5,2]
LATER_NEAR_USE_ENE = True
LATER_NEAR_USE_FRI = True

# t秒後のx,y,s
LATER_T = [0.5,1]
LATER_X = False
LATER_Y = False
LATER_HS = True
LATER_VS = True
LATER_S = False

# T later rdist
LATER_RDIST_TIME = [0.5,1,1.5,2]

# rusherとプレイヤーの線分が衝突するか　始点：現在地 -> Dir方向にSPEED*CONT
# later0が現在 speed使用しない
CROSS_CONT_LIS_CONST = [5,100]
CROSS_TIME_CONST = [0,0.5,1,1.5]
USE_REAL_VAL_CONST = True

# Tlater udist
UDIST_TIME = [0,0.5]
UDIST_PERSON = [0] #21はrusherなので不要c

# 近いユニットまでの距離
NEAREST_X = [0,1,2]
NEAREST_X_TIME = [0,0.5,1] 
NEAREST_X_FRI = False
NEAREST_X_ENE = True


def integrate_team_names(df):
    map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
    for abb in df['PossessionTeam'].unique():
        map_abbr[abb] = abb
    df['PossessionTeam'] = df['PossessionTeam'].map(map_abbr)
    df['FieldPosition'] = df['FieldPosition'].map(map_abbr)
    df['HomeTeamAbbr'] = df['HomeTeamAbbr'].map(map_abbr)
    df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].map(map_abbr)
    return df

def is_offence(df):
    df['is_offence'] = 'home'
    df.loc[df.PossessionTeam != df.HomeTeamAbbr, 'is_offence'] = "away"
    df['is_offence'] = (df.Team == df.is_offence).astype(np.uint8) # Is player on offense?
    return df

# 方向に合わせて標準化
def std_corr(df):
    df['ToLeft'] = df.PlayDirection == "left"
    df['YardLine_std'] = 100 - df.YardLine
    df.loc[df.FieldPosition.fillna('') == df.PossessionTeam,  
        'YardLine_std'
         ] = df.loc[df.FieldPosition.fillna('') == df.PossessionTeam,  
          'YardLine']
    df['X_std'] = df.X
    df.loc[df.ToLeft, 'X_std'] = 120 - df.loc[df.ToLeft, 'X'] 
    df['Y_std'] = df.Y
    df.loc[df.ToLeft, 'Y_std'] = 160/3 - df.loc[df.ToLeft, 'Y'] 
    df['Orientation_std'] = df.Orientation
    df.loc[df.ToLeft, 'Orientation_std'] = np.mod(180 + df.loc[df.ToLeft, 'Orientation_std'], 360)
    df['Dir_std'] = df.Dir
    df.loc[df.ToLeft, 'Dir_std'] = np.mod(180 + df.loc[df.ToLeft, 'Dir_std'], 360)
    cols = ['YardLine', 'X', 'Y', 'Orientation', 'Dir']
    for c in cols:
        df[c] = df[f'{c}_std']
    df.drop(['ToLeft']+[f'{c}_std' for c in cols], axis=1, inplace=True)
    return df

def is_rusher(df):
    df['is_rusher'] = (df['NflId'] == df['NflIdRusher']).astype(np.uint8)
    return df


def standard_preprocess(df):    
    df = integrate_team_names(df)
    df = is_offence(df)
    df = std_corr(df)
    df = is_rusher(df)
    resy = (100- df['YardLine']).values[::22].copy() #resy:は1
    resy0 = (99 - df['YardLine']).values[::22].copy() #:resy0は0
    return df, resy, resy0

def dist_rush_yard(df):
    playid2rushx = df[df['is_rusher']==1][['PlayId', 'X']].set_index('PlayId').to_dict()['X']
    df['dist_rush_yard'] = df['PlayId'].map(playid2rushx) - df['YardLine']
    return df

def get_dx_dy(radian_angle, dist):
    dx = dist * np.cos(radian_angle)
    dy = dist * np.sin(radian_angle)
    return dx, dy

def make_xy_diff(df):
    dfx = df['YardLine']
    dfy = (160/3)/2
    df['X'] = df['X']-dfx
    df['Y'] = df['Y']-dfy
    df['YardLine'] = df['YardLine'] - dfx
    dfx = df[df['is_rusher']==1].X.values
    dfy = df[df['is_rusher']==1].Y.values
    dfx = np.array([np.array([dfx[i]]*22) for i in range(dfx.shape[0])]).flatten()
    dfy = np.array([np.array([dfy[i]]*22) for i in range(dfy.shape[0])]).flatten()
    df['rdist'] = ((df['X'] - dfx)**2 + (df['Y'] - dfy)**2)**0.5
    return df

def sort_rows(df):
    df = df.sort_values(by=SORT_ORDER)
    return df

def angle_rad(df):
    df['Dir_rad'] = np.mod(90-df.Dir, 360) * math.pi/180.0
    df['Ori_rad'] = np.mod(90-df.Orientation, 360) * math.pi/180.0
    return df

def hv_speed(df):
    xsp, ysp = get_dx_dy(np.array(df.Dir_rad), np.array(df.S))
    df['h_speed'] = xsp
    df['v_speed'] = ysp 
    return df

def speed_relative_rusher(df):
    runner_sitas = np.array([[x]*22 for x in df[df['is_rusher']==1].Dir_rad]).flatten()
    df['sita_diff_from_runner'] = df['Dir_rad'] - runner_sitas
    df['speed_to_rusher'] = np.cos(df['sita_diff_from_runner'])*df['S']
    df['spped_vert_rusher'] = np.sin(df['sita_diff_from_runner'])*df['S']
    return df



def form_std(df):
    x_std = np.array([[df[i:i+11].X.std()]*11 for i in range(0,df.shape[0], 11)]).flatten()
    y_std = np.array([[df[i:i+11].Y.std()]*11 for i in range(0,df.shape[0], 11)]).flatten()
    df['x_stdev'] = x_std
    df['y_stdev'] = y_std
    return df

def later_from_std(df, ts=LATER_STD_T):
    xsp, ysp = get_dx_dy(np.array(df.Dir_rad), np.array(df.A))
    h_accel = xsp
    v_accel = ysp
    for t in ts:
        future_x = df['X'] + df['h_speed']*t + 0.5*h_accel*(t**2)
        future_y = df['Y'] + df['v_speed']*t + 0.5*v_accel*(t**2)
        x_std = np.array([[future_x[i:i+11].std()]*11 for i in range(0,df.shape[0], 11)]).flatten()
        y_std = np.array([[future_y[i:i+11].std()]*11 for i in range(0,df.shape[0], 11)]).flatten()
        df[f'later_x_stdev{t}'] = x_std
        df[f'later_y_stdev{t}'] = y_std
    return df
        
def num_player_near_sub(dfx, dfy ,fri, ene, place, th):
    for i in range(0,11):
        for j in range(i+1,11):
            if ((dfx[i]-dfx[j])**2 + (dfy[i]-dfy[j])**2)**0.5 <= th:
                fri[place+i] += 1
                fri[place+j] += 1
                
    for i in range(11,22):
        for j in range(i+1,22):
            if ((dfx[i]-dfx[j])**2 + (dfy[i]-dfy[j])**2)**0.5 <= th:
                fri[place+i] += 1
                fri[place+j] += 1
                
    for i in range(0,11):
        for j in range(11,22):
            if ((dfx[i]-dfx[j])**2 + (dfy[i]-dfy[j])**2)**0.5 <= th:
                ene[place+i] += 1
                ene[place+j] += 1

def num_player_near(df, ths=NEAR_YARD):
    for th in ths:
        res_fri = np.zeros(df.shape[0])
        res_ene = np.zeros(df.shape[0])
        for i in range(0, df.shape[0], 22):
            num_player_near_sub(df[i:i+22].X.values, df[i:i+22].Y.values, res_fri, res_ene, i, th)
        if NEAR_USE_FRI:
            df[f'num_fri{th}'] = res_fri
        if NEAR_USE_ENE:
            df[f'num_ene{th}'] = res_ene
    return df


def angle_0to180(df):
    df['dir0to180'] = np.mod(df['Dir']-90, 360)
    df.loc[df['dir0to180']>180,'dir0to180'] = 360 - df[df['dir0to180']>180]['dir0to180']
    df['ori0to180'] = np.mod(df['Orientation']-90, 360)
    df.loc[df['ori0to180']>180,'ori0to180'] = 360 - df[df['ori0to180']>180]['ori0to180']
    return df

def def_vs_dis(df):
    df['DefendersInTheBox_vs_Distance'] = df['DefendersInTheBox'] / df['Distance']
    return df

def add_later(df, ts=LATER_T):
    for t in ts:
        xsp, ysp = get_dx_dy(np.array(df.Dir_rad), np.array(df.A))
        h_accel = xsp
        v_accel = ysp
        if LATER_X:
            df[f'{t}later_x'] = df['X'] + df['h_speed']*t + 0.5*h_accel*(t**2)
        if LATER_Y:
            df[f'{t}later_y'] = df['Y'] + df['v_speed']*t + 0.5*v_accel*(t**2)
        if LATER_HS:
            df[f'{t}later_h_speed'] = df['h_speed'] + h_accel*t
        if LATER_VS:
            df[f'{t}later_v_speed'] = df['v_speed'] + v_accel*t
        if LATER_S:
            df[f'{t}later_speed'] = df['S'] + df['A']*t
        return df

def later_near(df, near=LATER_NEAR_NEAR ,ts=LATER_NEAR_T):
    xsp, ysp = get_dx_dy(np.array(df.Dir_rad), np.array(df.A))
    h_accel = xsp
    v_accel = ysp
    for t in ts:
        future_x = df['X'] + df['h_speed']*t + 0.5*h_accel*(t**2)
        future_y = df['Y'] + df['v_speed']*t + 0.5*v_accel*(t**2)
        for th in near:
            res_fri = np.zeros(df.shape[0])
            res_ene = np.zeros(df.shape[0])
            for i in range(0, df.shape[0], 22):
                num_player_near_sub(future_x[i:i+22].values, future_y[i:i+22].values, res_fri, res_ene, i, th)
            if LATER_NEAR_USE_FRI:
                df[f'later{t}_num_fri{th}'] = res_fri
            if LATER_NEAR_USE_ENE:
                df[f'later{t}num_ene{th}'] = res_ene
    return df

def later_rdist(df, ts=LATER_RDIST_TIME):
    xsp, ysp = get_dx_dy(np.array(df.Dir_rad), np.array(df.A))
    h_accel = xsp
    v_accel = ysp
    dfrush = df.loc[df['is_rusher']==1,:].copy()
    xsp, ysp = get_dx_dy(np.array(dfrush.Dir_rad), np.array(dfrush.A))
    h_accel_rush = xsp
    v_accel_rush = ysp
    for t in ts:
        future_x = df['X'] + df['h_speed']*t + 0.5*h_accel*(t**2)
        future_y = df['Y'] + df['v_speed']*t + 0.5*v_accel*(t**2)
        dfx = (dfrush['X'] + dfrush['h_speed']*t + 0.5*h_accel_rush*(t**2)).values
        dfy = (dfrush['Y'] + dfrush['v_speed']*t + 0.5*v_accel_rush*(t**2)).values
        dfx = np.array([np.array([dfx[i]]*22) for i in range(dfx.shape[0])]).flatten()
        dfy = np.array([np.array([dfy[i]]*22) for i in range(dfy.shape[0])]).flatten()
        diffx = future_x - dfx
        diffy = future_y - dfy
        df[f'later_rdist_{t}'] = (diffx**2 + diffy**2)**0.5
    return df

def later_udist(df, ts=UDIST_TIME, center=UDIST_PERSON):
    xsp, ysp = get_dx_dy(np.array(df.Dir_rad), np.array(df.A))
    h_accel = xsp
    v_accel = ysp
    for c in center:
        dfrush = df[c::22].copy()
        xsp, ysp = get_dx_dy(np.array(dfrush.Dir_rad), np.array(dfrush.A))
        h_accel_rush = xsp
        v_accel_rush = ysp
        for t in ts:
            future_x = df['X'] + df['h_speed']*t + 0.5*h_accel*(t**2)
            future_y = df['Y'] + df['v_speed']*t + 0.5*v_accel*(t**2)
            dfx = (dfrush['X'] + dfrush['h_speed']*t + 0.5*h_accel_rush*(t**2)).values
            dfy = (dfrush['Y'] + dfrush['v_speed']*t + 0.5*v_accel_rush*(t**2)).values
            dfx = np.array([np.array([dfx[i]]*22) for i in range(dfx.shape[0])]).flatten()
            dfy = np.array([np.array([dfy[i]]*22) for i in range(dfy.shape[0])]).flatten()
            diffx = future_x - dfx
            diffy = future_y - dfy
            df[f'udist_time{t}_ceterperson{c}'] = (diffx**2 + diffy**2)**0.5
    return df

def nearest_X(df, xs=NEAREST_X,ts=NEAREST_X_TIME):
    df_temp = df.copy()
    df_temp = later_udist(df_temp, ts, [i for i in range(0,22)])
    for t in ts:
        nearest_off = df_temp[[f'udist_time{t}_ceterperson{c}' for c in [i for i in range(11,22)]]].values.copy()
        nearest_off.sort(axis=1)
        nearest_dif = df_temp[[f'udist_time{t}_ceterperson{c}' for c in [i for i in range(0,11)]]].values.copy()
        nearest_dif.sort(axis=1)
        for x in xs:
            if NEAREST_X_ENE:
                df[f'nearest_ene_{x}_{t}'] = nearest_dif[:,x]
                off_temp = df[f'nearest_ene_{x}_{t}'].copy()
                df[f'nearest_ene_{x}_{t}'] = nearest_off[:,x]
                df.loc[df['is_offence']==1,f'nearest_ene_{x}_{t}'] = off_temp
            if NEAREST_X_FRI:
                # friの最近は自分
                df[f'nearest_fri_{x}_{t}'] = nearest_off[:,x+1]
                off_temp = df[f'nearest_fri_{x}_{t}'].copy()
                df[f'nearest_fri_{x}_{t}'] = nearest_dif[:,x+1]
                df.loc[df['is_offence']==1,f'nearest_fri_{x}_{t}'] = off_temp
    return df


def back_direction(orientation):
    if orientation > 180.0:
        return 1
    else:
        return 0



# http://www5d.biglobe.ne.jp/~tomoya03/shtml/algorithm/Intersection.htm
def rusher_cross_line_const(df, rs=CROSS_CONT_LIS_CONST, ts=CROSS_TIME_CONST, use_real=USE_REAL_VAL_CONST):
    xsp, ysp = get_dx_dy(np.array(df.Dir_rad), np.array(df.A))
    h_accel = xsp
    v_accel = ysp
    dfrush = df.loc[df['is_rusher']==1,:].copy()
    xsp, ysp = get_dx_dy(np.array(dfrush.Dir_rad), np.array(dfrush.A))
    h_accel_rush = xsp
    v_accel_rush = ysp
    for t in ts:
        future_x = df['X'] + df['h_speed']*t + 0.5*h_accel*(t**2)
        future_y = df['Y'] + df['v_speed']*t + 0.5*v_accel*(t**2)
        future_s = (df['S'] + df['A']*t).values
        dfx = (dfrush['X'] + dfrush['h_speed']*t + 0.5*h_accel_rush*(t**2)).values
        dfy = (dfrush['Y'] + dfrush['v_speed']*t + 0.5*v_accel_rush*(t**2)).values
        dfs = (dfrush['S'] + dfrush['A']*t).values

        df_rush = df.loc[df['is_rusher']==1, :].copy()
        for R in rs:
            R_rusher = R
            R_player = R

            x1 = future_x.copy().values
            y1 = future_y.copy().values
            x2 = x1 + np.cos(df['Dir_rad'].copy().values)*R_player
            y2 = y1 + np.sin(df['Dir_rad'].copy().values)*R_player
            x3 = dfx.copy()
            y3 = dfy.copy()
            x4 = x3 + np.cos(df_rush['Dir_rad'].copy().values)*R_rusher
            y4 = y3 + np.sin(df_rush['Dir_rad'].copy().values)*R_rusher
            x3 = np.array([np.array([x3[i]]*22) for i in range(x3.shape[0])]).flatten()
            y3 = np.array([np.array([y3[i]]*22) for i in range(y3.shape[0])]).flatten()
            x4 = np.array([np.array([x4[i]]*22) for i in range(x4.shape[0])]).flatten()
            y4 = np.array([np.array([y4[i]]*22) for i in range(y4.shape[0])]).flatten()

            ta = (x3-x4)*(y1-y3) + (y3-y4)*(x3-x1)
            tb = (x3-x4)*(y2-y3) + (y3-y4)*(x3-x2)
            tc = (x1-x2)*(y3-y1) + (y1-y2)*(x1-x3)
            td = (x1-x2)*(y4-y1) + (y1-y2)*(x1-x4)
            with np.errstate(invalid='ignore'):
                res = (ta*tb<0).astype(np.uint8)
                res2 = (tc*td<0).astype(np.uint8)
            if use_real:
                df[f'cross_line_cont{R}_later{t}_res1_CONST'] = ta*tb
                df[f'cross_line_cont{R}_later{t}_res2_CONST'] = tc*td
            df[f'cross_line_cont{R}_later{t}_CONST'] = (res+res2 == 2).astype(np.uint8)
    return df



 # LATER_RDIST_TIME,CROSS_TIME_CONSの共通
def new_rdist(df, ts=NEW_R_TIME):
    for t in ts:
        df[f'newR_{t}'] = df[f'later_rdist_{t}']+5*(1-df[f'cross_line_cont100_later{t}_CONST'])
        df[f'newR2_{t}'] = df[f'later_rdist_{t}']+5*(1-df[f'cross_line_cont100_later{t}_CONST'])+5*df['is_offence']
        #df.drop(f'cross_line_cont100_later{t}_CONST',axis=1,inplace=True)
    return df

def feature_eng(df):
    # scremearge lineとrusherの距離
    df = dist_rush_yard(df) 
    df = make_xy_diff(df) # #座標の中心を取り直す
    df = sort_rows(df)
    df = angle_rad(df)
    df = hv_speed(df)
    df = speed_relative_rusher(df)
    df = form_std(df)
    df = later_from_std(df)
    df = num_player_near(df)
    df = later_near(df)
    df = angle_0to180(df)
    df = def_vs_dis(df)
    df = add_later(df)
    df = later_rdist(df)
    df = rusher_cross_line_const(df)
    df = nearest_X(df)
    df = new_rdist(df)
    return df

def del_some_cols(df):
    del_cols = ['GameId','PlayId','NflId', 'JerseyNumber', 'Season', 'YardLine','GameClock',
               'HomeScoreBeforePlay', 'VisitorScoreBeforePlay','NflIdRusher', 
               'Week', 'Temperature', 'Humidity','HomePossesion', 'Field_eq_Possession',
               'TimeDelta', 'PlayerAge', 'Grass', 'PlayerHeight']
    del_cols += ['Team', 'DisplayName', 'PossessionTeam','FieldPosition','PlayDirection', 'TimeHandoff', 'TimeSnap',
                 'PlayerBirthDate', 'PlayerCollegeName','HomeTeamAbbr', 'VisitorTeamAbbr', 'Stadium', 'Location', 'StadiumType', 'Turf', 'GameWeather', 'WindSpeed', 'WindDirection']

    return df.drop(del_cols, axis=1, errors='ignore')

def padNan_std(df, scaler, numcol2median):
    df = df.fillna(numcol2median)
    #df = df.fillna(-999)
    cols = df.columns

    df = scaler.transform(df)
    df = pd.DataFrame(df)
    df.columns = cols
    return df

def padNan_cat(df):
    df = df.fillna(-1)
    return df


def sep_cols(df, num_features, cat_features):
    return [df[num_features], df[cat_features]]

def expand_data(df):
    retX = []
    for i in range(0, df.shape[0], 22):
        retX.append(df[i:i+22].values)
    return np.array(retX)