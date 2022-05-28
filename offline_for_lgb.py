# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 09:20:07 2021

@author: hp
"""

import pandas as pd
import random
import numpy as np
import lightgbm as lgb


data = pd.read_csv('prepare/train_for_tree.csv')
df_train = data[data['date_']<14]
df_test = data[data['date_']==14]

feats_all = [x for x in df_train.columns if x not in ['userid','feedid','authorid','device','date_','read_comment',
                                                  'like','click_avatar','forward','ua_id','bgm_singer_id','bgm_song_id']]

df_train[feats_all] = df_train[feats_all].fillna(0)
df_test[feats_all] = df_test[feats_all].fillna(0)

clf = lgb.LGBMClassifier(learning_rate=0.01,max_depth=6,n_estimators=2000,random_state=2020,num_leaves=64,
                         n_jobs=22,subsample=0.8,subsample_freq=5,colsample_bytree=0.8)

sample_0 = df_train[df_train['read_comment']==0].sample(len(df_train[df_train['read_comment']==1])*10,random_state=2020)
sample = sample_0.append(df_train[df_train['read_comment']==1])
clf.fit(sample[feats_all],sample['read_comment'])
y_pred = clf.predict_proba(df_test[feats_all])[:,1]
df_test['read_comment_prob'] = y_pred
clf.booster_.save_model("offline_lgb_rc.txt")

sample_0 = df_train[df_train['like']==0].sample(len(df_train[df_train['like']==1])*10,random_state=2020)
sample = sample_0.append(df_train[df_train['like']==1])
clf.fit(sample[feats_all],sample['like'])
y_pred = clf.predict_proba(df_test[feats_all])[:,1]
df_test['like_prob'] = y_pred
clf.booster_.save_model("offline_lgb_like.txt")

sample_0 = df_train[df_train['click_avatar']==0].sample(len(df_train[df_train['click_avatar']==1])*10,random_state=2020)
sample = sample_0.append(df_train[df_train['click_avatar']==1])
clf.fit(sample[feats_all],sample['click_avatar'])
y_pred = clf.predict_proba(df_test[feats_all])[:,1]
df_test['click_avatar_prob'] = y_pred
clf.booster_.save_model("offline_lgb_ca.txt")

sample_0 = df_train[df_train['forward']==0].sample(len(df_train[df_train['forward']==1])*10,random_state=2020)
sample = sample_0.append(df_train[df_train['forward']==1])
clf.fit(sample[feats_all],sample['forward'])
y_pred = clf.predict_proba(df_test[feats_all])[:,1]
df_test['forward_prob'] = y_pred
clf.booster_.save_model("offline_lgb_forward.txt")

res_df = df_test[['userid','feedid','read_comment_prob','like_prob','click_avatar_prob','forward_prob']]
res_df = res_df.rename(columns={'read_comment_prob':'read_comment','like_prob':'like','click_avatar_prob':'click_avatar',
                      'forward_prob':'forward'})
res_df.to_csv('submit_offline_lgb.csv',index=False)


