# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 09:41:40 2021

@author: hp
"""


import pandas as pd
import catboost as cat
import random
import numpy as np

df_train = pd.read_csv('../../data/train_for_tree.csv')

feats_all = [x for x in df_train.columns if x not in ['userid','feedid','authorid','device','date_','read_comment',
                                                  'like','click_avatar','forward','ua_id']]

df_train[feats_all] = df_train[feats_all].fillna(0)

cbt = cat.CatBoostClassifier(iterations=3000,learning_rate=0.05,depth=6,verbose=False,
                             random_seed=2020,num_leaves=64
                             ,colsample_bylevel=0.8,subsample=0.8,thread_count=22,eval_metric='AUC')

sample_0 = df_train[df_train['read_comment']==0].sample(len(df_train[df_train['read_comment']==1])*10,random_state=2020)
sample = sample_0.append(df_train[df_train['read_comment']==1])
cbt.fit(sample[feats_all],sample['read_comment'])
cbt.save_model('online_cbt_rc.model')

sample_0 = df_train[df_train['like']==0].sample(len(df_train[df_train['like']==1])*10,random_state=2020)
sample = sample_0.append(df_train[df_train['like']==1])
cbt.fit(sample[feats_all],sample['like'])
cbt.save_model('online_cbt_like.model')

sample_0 = df_train[df_train['click_avatar']==0].sample(len(df_train[df_train['click_avatar']==1])*10,random_state=2020)
sample = sample_0.append(df_train[df_train['click_avatar']==1])
cbt.fit(sample[feats_all],sample['click_avatar'])
cbt.save_model('online_cbt_ca.model')

sample_0 = df_train[df_train['forward']==0].sample(len(df_train[df_train['forward']==1])*10,random_state=2020)
sample = sample_0.append(df_train[df_train['forward']==1])
cbt.fit(sample[feats_all],sample['forward'])
cbt.save_model('online_cbt_forward.model')

