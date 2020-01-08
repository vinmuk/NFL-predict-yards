import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import keras
import os
import sys
import tensorflow as tf
from keras import backend as K
from keras import Input, Model
from keras.layers import Dense, Flatten, Concatenate,Embedding,Reshape,Dropout,Lambda,Activation,BatchNormalization,ReLU
from keras.regularizers import l2
from gcn_layer import *
from preprocess import *
from util import *
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import RepeatedKFold, GroupKFold
from keras.optimizers import Adam
import gc
args = sys.argv
TRAIN_PATH = args[1]
TEST_PATH = args[2]

# model parameters
DROP_RATE = 0.2
INITIAL_LR = 1e-3

# objectもしくは nuniqueが下記の値!未満!のfeatのみcatとして扱う デフォ：20
ORDER_CAT_NO_EMB_TH = 10
# embサイズ　基本は入力のEMB_C 倍, 上限は EMB_MAX
EMB_C = 100
EMB_MAX = 1
l2_reg = 5e-4 
all_val_scores = []
all_val_pi_scores = {}

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(7)
"""
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=0,
    inter_op_parallelism_threads=0
)
tf.set_random_seed(8)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
"""
start = time.time()

# adjacency matrix
# rusher-ene, i番目は i-x,i,i+1番目のeneとedge 
half_side_num_player = 2
adj = np.zeros((22,22))
for i in range(0,21):
    adj[i,21] = adj[21,i] = 1
for i in range(0,11):
    for j in range(max(11,i+11-half_side_num_player),min(21,i+11+half_side_num_player+1)):
        adj[i,j] = adj[j,i] = 1

# import train data and verify it
train = pd.read_csv(TRAIN_PATH, dtype={'WindSpeed': 'object'})
test = pd.read_csv(TEST_PATH, dtype={'WindSpeed': 'object'})

# verify train data
try:
    # some fix
    print(train.shape)
    train = train.loc[~train['Yards'].isna(),:]
    train['Yards'] = np.clip(train['Yards'].astype(np.int16),-99,99)
    
    print(train.shape)
    train['X'] = np.clip(train['X'], 0, 120)
    train['Y'] = np.clip(train['Y'], 0, 60)
    train['Dir'] = np.mod(train['Dir'], 361)
    train['Orientation'] = np.mod(train['Orientation'], 361)
    print(train.shape)
    # each play has 22 players
    ct = train.groupby('PlayId')['Yards'].transform('size')
    train = train.loc[ct==22,:]
    print(train.shape)
    # add new gameid whose gameid is null
    # each play has a rusher
    norush_play = train[train['NflId']==train['NflIdRusher']].set_index('PlayId').groupby('PlayId')['Yards'].transform('size').to_dict()
    c = train['PlayId'].map(norush_play)
    train = train.loc[c==1,:]
    print(train.shape)
except:
    train = pd.read_csv(TRAIN_PATH, dtype={'WindSpeed': 'object'})
    
min_yards = train.Yards.min()

# output when inference failed
bad_output = np.histogram(train["Yards"], bins=199,
                 range=(-99,100), density=True)[0].cumsum()


# feature engineering
train, res_yards, resy_0 = standard_preprocess(train)
train = feature_eng(train)

# Augment data and reindex playid
train = augment(train)
# conctしてindexが重複&PlayIDがaugとorgで重複してはならない
train = train.reset_index().drop('index', axis=1)
train['PlayId'] = np.array([[i]*22 for i in range(0, int(train.shape[0]/22))]).flatten()
train_enged = train.copy()

# Make target data
y_train50 = split_y(train, 50)
y_mae = train['Yards'][::22].values.copy()
y_train = np.zeros(shape=(int(train.shape[0]/22), 199))
for i,yard in enumerate(train['Yards'][::22]):
    y_train[i, yard+99:] = np.ones(shape=(1, 100-yard))  
train.drop('Yards', axis=1, inplace=True)

# delete unnecessary features
train = del_some_cols(train)

# Categorical feats and numerical feats
cat_features = []
num_features = []
for col in train.columns:
    if train[col].dtype =='object' or train[col].nunique()<ORDER_CAT_NO_EMB_TH:
        cat_features.append(col)
        #cat_features.append((col, len(train[col].unique())))
    else:
        num_features.append(col)

# separate train data for numerical and categorical
train_num, train_cat = sep_cols(train, num_features, cat_features)

# standarize and fillna
scaler = StandardScaler() 
scaler.fit(train_num) 
# nan values are padding with median
numcol2median = {}
numcol2max = {}
for c in train_num.columns:
    numcol2median[c] = train_num[c].median()
    numcol2max[c] = train_num[c].max()
train_num = padNan_std(train_num, scaler, numcol2median)
train_cat = padNan_cat(train_cat)

# label encoding
les = []
for col in train_cat.columns:
    le = LabelEncoder()
    le.fit(train_cat[col].astype(str))
    les.append(le)
for i, col in enumerate(train_cat.columns):
    le = les[i]
    train_cat[col] = le.transform(train_cat[col].astype(str))

# expand data
train_num = expand_data(train_num)
train_cat = expand_data(train_cat)

N = 22
F = train_num.shape[2]
group = train_enged['GameId'][::22].copy()
group = group.fillna(9029)


def train_M3(x_tr, y_tr, x_vl, y_vl, y_tr5, y_vl5, ymtr, ymvl):    
    inputs = []
    embeddings = []
    for i in range(len(cat_features)):
        input_ = Input(shape=(22,))
        content_num = np.absolute(train_cat[:,:,i]).max()
        esize = int(min(content_num*EMB_C, EMB_MAX))
        embedding = Embedding(int(content_num + 1), esize, input_length=22)(input_)
        inputs.append(input_)
        embeddings.append(embedding)
    x = Concatenate()(embeddings)
    X_in = Input(shape=(N, F))
    X_in2 = Concatenate()([X_in, x])

    A_in = Input(tensor=K.constant(adj))
    inputs.append(X_in)
    inputs.append(A_in)
    graph_conv = GraphAttention2(16,
                                 attn_heads=4,
                           attn_heads_reduction='concat',
                           activation='elu',
                           kernel_regularizer=l2(l2_reg),
                           use_bias=True)([X_in2, A_in])
    #fc0 = Flatten()(graph_conv)
    fc0 = GlobalAttnSumPool()(graph_conv)
    pre_output1 = Dense(50, activation='linear')(fc0)
    output1 = Activation('softmax')(pre_output1)
    graph_conv = GraphAttention2(16,
                                 attn_heads=4,
                           attn_heads_reduction='average',
                           activation='elu',
                           kernel_regularizer=l2(l2_reg),
                           use_bias=True)([graph_conv, A_in])
    flatten = GlobalAttnSumPool()(graph_conv)
    #flatten = Flatten()(graph_conv)
    pre_output2 = Dense(50, activation='linear')(flatten)
    output2 = Activation('softmax')(pre_output2)
    fc = Concatenate()([fc0,flatten])
    mae_out = Dense(1, activation=None)(fc)

    output = Dense(199, activation='sigmoid', name='org')(fc)
    
    # Build model
    model = Model(inputs=inputs, outputs=[output2, output1, mae_out, output])
    
    er = EarlyStopping(patience=5, min_delta=1e-5, restore_best_weights=True, monitor='val_org_loss')
    reduce_lr = ReduceLROnPlateau(monitor='val_org_loss', factor=0.5, mode='min',patience=5, verbose=1)
    checkpoint = ModelCheckpoint('temp.h5', monitor='val_org_loss', verbose=1, save_best_only=True, save_weights_only=True)

    model.compile(optimizer=Adam(lr=INITIAL_LR), loss=['categorical_crossentropy','categorical_crossentropy','mae','mse'], loss_weights=[1,1,1,350])

    model.fit(x_tr, [y_tr5,y_tr5, ymtr,y_tr], epochs=100, callbacks=[er,reduce_lr,checkpoint], validation_data=[x_vl, [y_vl5, y_vl5,ymvl, y_vl]], verbose=0)
    model.load_weights('temp.h5')
    val_pred = model.predict(x_vl)[-1]

    all_val_scores.append(crps(y_vl, val_pred))
    return model

# main.py
def make_pred_df_process(df):
    df = del_some_cols(df)
    df_num, df_cat = sep_cols(df, num_features, cat_features)
    df_num = padNan_std(df_num, scaler, numcol2median)
    df_cat = padNan_cat(df_cat)
    for i, col in enumerate(df_cat.columns):
        try:
            le = les[i]
            df_cat[col] = le.transform(df_cat[col].astype(str))
        except:
            subst = train_enged[col][0]
            df_cat.loc[~df_cat[col].isin(set(train_enged[col])),col] = subst
            le = les[i]
            df_cat[col] = le.transform(df_cat[col].astype(str))
    df_num = expand_data(df_num)
    df_cat = expand_data(df_cat)
    df = [df_cat[:,:,i] for i in range(df_cat.shape[-1])] + [df_num]
    return df


def make_predNN(df,models):
    df['DefendersInTheBox'] = df['DefendersInTheBox'].astype(np.float64) 
    df,res_yards,res_yards0 = standard_preprocess(df)
    df = feature_eng(df)
    df = make_pred_df_process(df)
    y_pred = np.mean([model.predict(df)[-1] for model in models], axis=0)
    for pred in y_pred:
        prev = 0
        for i in range(len(pred)):
            if pred[i]<prev:
                pred[i]=prev
            prev=pred[i]
    # post process
    for i, pred in enumerate(y_pred):
        pred[res_yards[i]+99:] = 1
        pred[:res_yards0[i]] = 0
        pred[:min_yards+99] = 0
    return y_pred

def make_pred(df, models):
    try:
        y_pred = make_predNN(df,models)
    except:
        y_pred = np.array([bad_output]*int(df.shape[0]/22))
    return y_pred

rkf = GroupKFold(n_splits=10)
fold_start = time.time()
models = []
X_train = train_num
count = 0
for tr_idx, vl_idx in rkf.split(X_train, y_train, group):
    gc.collect()
    
    if count == 0:
        t = 0
    else:
        t = time.time()-start + (time.time()-fold_start)/count
    
    print(f'current passed time:{time.time()-start}, estimated next fold time:{t}')
    if t >=8500:
        break
        
    count += 1
    x_num_tr,x_num_vl  = train_num[tr_idx], train_num[vl_idx]
    x_cat_tr,x_cat_vl  = train_cat[tr_idx], train_cat[vl_idx]
    y_tr, y_vl = y_train[tr_idx], y_train[vl_idx]
    ymtr, ymvl = y_mae[tr_idx], y_mae[vl_idx]
    Xt = [x_cat_tr[:,:,i] for i in range(train_cat.shape[-1])] + [x_num_tr]
    Xv = [x_cat_vl[:,:,i] for i in range(train_cat.shape[-1])] + [x_num_vl]
    y_tr5, y_vl5 = y_train50[tr_idx], y_train50[vl_idx]
    model = train_M3(Xt,y_tr, Xv, y_vl, y_tr5, y_vl5,ymtr, ymvl)
    models.append(model)

print(all_val_scores)
print(np.mean(all_val_scores))

y_pred =  make_pred(test, models)
y_pred = pd.DataFrame(y_pred)
y_pred.to_csv('result.csv', index=False)