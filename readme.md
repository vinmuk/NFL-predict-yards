# NFL predict yards

7th place code in the competition at https://www.kaggle.com/c/nfl-big-data-bowl-2020/overview

Purpose of this competition is predicting how many yards will an rusher gain in each running play. Tracking data at handoff for each player is used to predict.

GCN Layer codes are heavily depends on https://github.com/danielegrattarola/spektral

# Overview
- Data fix
  - No data modification.
  - With some features, there was a difference in distribution, however, in my case, fix them by adjusting mean and standard deviation makes score worse.

- Model
  - 2-layer GCN with multi output.
  - Attention sum pooling layer (attention is computed by dot products of trainable weights and features of each player) follows each GCN layer 
  - GCN layer is my original layer which combine Graph attention networks (GAT) and GraphConvSkip layer.
  

- Optimizer
  - Adam

- Loss function
  - Binary crossentropy and mae for last layer
  - Binary crossentropy loss with divided output into 50 sections are located immediately after each GCN layer. This hastens the time to convergence.

- Data Augment
  - Flip y axis
  - Add data after a few seconds doesn't work
  - TTA doesn't work

- Feature engineering
    - 115 features for each player
    - Separate player feature and play feature made score worse
    -  Below features after 0~2 seconds significantly improved score.
       -  the number of opponent players around each player.
       -  whether to collide with rusher when traveling a certain distance
       -  distance from rusher

# How to run
```
$python train_predict.py TRAIN_CSV_PATH TEST_CSV_PATH
```
