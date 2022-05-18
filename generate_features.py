# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 11:06:15 2021

@author: hp
"""


import numpy as np
import pandas as pd

feedinfo = pd.read_csv("../../data/wechat/wechat_algo_data1/feed_info.csv")

print('----------generate user features----------')
def getNumFeatures(df_his,target_col,auxiliary_col,col_list, suffix):
    tmp_sta = df_his.groupby(target_col,as_index=False)[col_list].agg(set)
    col_list_new = [target_col + '_' + x+'_set' for x in col_list]
    tmp_sta.columns = [target_col] + col_list_new
    for each_feat in col_list_new:
        tmp_sta[ each_feat[:-4]+'_num'+suffix ] = tmp_sta[each_feat].apply(lambda x:len(x))
    
    count_sta = df_his[[target_col,auxiliary_col]].groupby([target_col],as_index=False).count()
    count_sta = count_sta.rename(columns={auxiliary_col:auxiliary_col+'_count'+suffix})
    tmp_sta = pd.merge(tmp_sta,count_sta,how='left',on=[target_col])
    
    return tmp_sta[[target_col,auxiliary_col+'_count'+suffix]+[x for x in tmp_sta.columns if '_num' in x]]

def getSumFeatures(df_his, target_col, col_list, df_tmp, suffix):
    tmp_sta = df_his.groupby(target_col,as_index=False)[col_list].sum()
    col_list_new = [target_col + '_' + x+'_sum'+suffix for x in col_list]
    tmp_sta = tmp_sta.rename(columns=dict(zip(col_list,col_list_new)))
    return tmp_sta

def getFeature(df, target_col,auxiliary_col,col_list_num,col_list_sum,tag):
    user_sta_1 = getNumFeatures(df,target_col,auxiliary_col,col_list_num,tag)
    user_sta_2 = getSumFeatures(df, target_col, col_list_sum, user_sta_1[auxiliary_col+'_count'+tag],suffix=tag)
    user_sta = pd.merge(user_sta_1,user_sta_2,how='left')
    
    return user_sta 

END_DAY = 15
start_day = 2

target_col = 'userid'
auxiliary_col = 'feedid'
col_list_num = ['feedid','authorid','date_','device']
col_list_sum = ['read_comment','comment','like','play','stay','click_avatar','forward','follow','favorite','behavior','behavior_flag']

for start in range(start_day, END_DAY+1):
    print(start)
    print('---')
    
    df_tmp = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    df_tmp = pd.merge(df_tmp,feedinfo[['feedid','authorid']],how='left')
    df_tmp['behavior'] = df_tmp [['read_comment','comment','like','click_avatar','forward','follow','favorite']].sum(axis=1)
    df_tmp['behavior_flag'] = df_tmp['behavior'].apply(lambda x: 1 if x>=1 else 0)
        
    tag = 'new'
    user_sta = getFeature(df_tmp, target_col,auxiliary_col,col_list_num,col_list_sum,tag)
    
    user_sta.to_csv('User_Features_'+str(start-1)+'.csv',index=False)

    
print('----------generate feed features----------')
END_DAY = 15
start_day = 2
target_col = 'feedid'
auxiliary_col = 'userid'
col_list_sum = ['read_comment','comment','like','play','stay','click_avatar','forward','follow','favorite','behavior','behavior_flag']
col_list_num = ['userid','date_','device']

for start in range(start_day, END_DAY+1):
    print(start)
    print('---')
    
    df_tmp = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    df_tmp['behavior'] = df_tmp [['read_comment','comment','like','click_avatar','forward','follow','favorite']].sum(axis=1)
    df_tmp['behavior_flag'] = df_tmp['behavior'].apply(lambda x: 1 if x>=1 else 0)
        
    tag = 'new'
    feed_sta = getFeature(df_tmp, target_col,auxiliary_col,col_list_num,col_list_sum,tag)
        
    feed_sta.to_csv('Feed_Features_'+str(start-1)+'.csv',index=False)


print('----------generate author features----------')
def getFeature(df,target_col,col_list_num,col_list_sum,tag):
    sta_1 = getNumFeatures(df,target_col,col_list_num,tag)

    sta_2 = getSumFeatures(df, target_col, col_list_sum, sta_1['authorid_feedid_num'+tag],suffix=tag)
    sta = pd.merge(sta_1,sta_2,how='left')
    
    return sta 

END_DAY = 15
start_day = 2
target_col = 'authorid'
col_list_num = ['userid','feedid','date_','device']
col_list_sum = ['read_comment','comment','like','play','stay','click_avatar','forward','follow','favorite']

for start in range(start_day, END_DAY+1):
    print(start)
    print('---')
    
    df_tmp = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    df_tmp = pd.merge(df_tmp,feedinfo[['feedid','authorid']],how='left')
    
    tag = 'new'
    author_sta = getFeature(df_tmp, target_col,col_list_num,col_list_sum,tag)
    
    author_sta.to_csv('Author_Features_'+str(start-1)+'.csv',index=False)

    
print('----------generate UserAuthor features----------')
END_DAY = 15
start_day = 2

for start in range(start_day, END_DAY+1):
    print(start)
    print('---')
    
    df_tmp = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    df_tmp = pd.merge(df_tmp,feedinfo[['feedid','authorid']],how='left')
    df_tmp['count'] = 1 
    
    user_author_sta = df_tmp.groupby(['userid','authorid'],as_index=False)['read_comment','comment','like','play',
                                                           'stay','click_avatar','forward','follow','favorite','count'].sum()
    
    col = [x for x in user_author_sta if x not in ['userid','authorid']]
    user_author_sta = user_author_sta.rename(columns=dict(zip(col,['ua_'+x for x in col])))
    
    user_author_sta.to_csv('User_Author_Interaction_Features_'+str(start-1)+'.csv',index=False)

    
print('----------generate Pro features----------')
import numpy as np
import random
import scipy.special as special
import math
from math import log
import pandas as pd


class HyperParam(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        sample = np.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return I, C

    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        for i in range(len(tries)):
            sumfenzialpha += (special.digamma(success[i] + alpha) - special.digamma(alpha))
            sumfenzibeta += (special.digamma(tries[i] - success[i] + beta) - special.digamma(beta))
            sumfenmu += (special.digamma(tries[i] + alpha + beta) - special.digamma(alpha + beta))

        return alpha * (sumfenzialpha / sumfenmu), beta * (sumfenzibeta / sumfenmu)

    def update_from_data_by_moment(self, tries, success):
        mean, var = self.__compute_moment(tries, success)
        self.alpha = (mean + 0.000001) * ((mean + 0.000001) * (1.000001 - mean) / (var + 0.000001) - 1)
        self.beta = (1.000001 - mean) * ((mean + 0.000001) * (1.000001 - mean) / (var + 0.000001) - 1)

    def __compute_moment(self, tries, success):
        ctr_list = []
        var = 0.0
        for i in range(len(tries)):
            ctr_list.append(float(success[i]) / tries[i])
        mean = sum(ctr_list) / len(ctr_list)
        for ctr in ctr_list:
            var += pow(ctr - mean, 2)
        return mean, var / (len(ctr_list) - 1)


def test():
    hyper = HyperParam(1, 1)
    I, C = hyper.sample_from_beta(10, 1000, 10000, 1000)
    print(I, C)
    print(I)
    print(len(I))
    print(C)
    print(len(C))
    hyper.update_from_data_by_moment(I, C)
    print(hyper.alpha, hyper.beta)

def ctr(I,C,alpha,beta):
    trade_slide = [i + j for i, j in zip(C, [alpha] * len(C))]
    clide_slide = [i + j + k for i, j, k in zip(I, [alpha] * len(I), [beta] * len(I))]
    return [i/j for i,j in zip(trade_slide, clide_slide)]

for i in range(1,15):
    df = pd.read_csv('User_Features_'+str(i)+'.csv')
    feats = [x for x in df.columns if 'day' not in x and 'avg' not in x]
    pro_col = [x for x in feats if 'sum' in x and 'play' not in x and 'stay' not in x and 'behavior_sum' not in x ]
    df_tmp = df[['userid','feedid_countnew']+pro_col]
    
    for each_feat in pro_col:
        hyper = HyperParam(1, 1)
        hyper.update_from_data_by_moment(df_tmp['feedid_countnew'], df_tmp[each_feat])
        df_tmp[each_feat+'_pro'] = ctr(df_tmp['feedid_countnew'], df_tmp[each_feat],hyper.alpha,hyper.beta)
    
    df_tmp[['userid']+[x+'_pro' for x in pro_col]].to_csv('User_Pro_Features_'+str(i)+'.csv')
    print(i)

for i in range(1,15):
    df = pd.read_csv('Feed_Features_'+str(i)+'.csv')
    feats = [x for x in df.columns if 'day' not in x and 'avg' not in x]
    pro_col = [x for x in feats if 'sum' in x and 'play' not in x and 'stay' not in x and 'behavior_sum' not in x ]
    df_tmp = df[['feedid','userid_countnew']+pro_col]
    
    for each_feat in pro_col:
        hyper = HyperParam(1, 1)
        hyper.update_from_data_by_moment(df_tmp['userid_countnew'], df_tmp[each_feat])
        df_tmp[each_feat+'_pro'] = ctr(df_tmp['userid_countnew'], df_tmp[each_feat],hyper.alpha,hyper.beta)
    
    df_tmp[['feedid']+[x+'_pro' for x in pro_col]].to_csv('Feed_Pro_Features_'+str(i)+'.csv')
    print(i)

for i in range(1,15):
    df = pd.read_csv('Author_Features_'+str(i)+'.csv')
    feats = [x for x in df.columns if 'day' not in x and 'avg' not in x]
    pro_col = [x for x in feats if 'sum' in x and 'play' not in x and 'stay' not in x and 'behavior_sum' not in x ]
    df_tmp = df[['authorid','authorid_countnew']+pro_col]
    
    for each_feat in pro_col:
        hyper = HyperParam(1, 1)
        hyper.update_from_data_by_moment(df_tmp['authorid_countnew'], df_tmp[each_feat])
        df_tmp[each_feat+'_pro'] = ctr(df_tmp['authorid_countnew'], df_tmp[each_feat],hyper.alpha,hyper.beta)
    
    df_tmp[['authorid']+[x+'_pro' for x in pro_col]].to_csv('Author_Pro_Features_'+str(i)+'.csv',index=False)
    print(i)
    break


print('----------generate MMS features--------------')
from gensim.models import KeyedVectors
import pandas as pd
import catboost as cat
import random
import numpy as np
import pickle
import gc
from tqdm import tqdm
from scipy import stats
MM_model = KeyedVectors.load_word2vec_format('feed_embeddings_new.csv')
END_DAY = 15
start_day = 2

for start in range(start_day, END_DAY+1):
    
    df_tmp_log = pd.read_csv('Sample_new/offline_test_'+'1_'+str(start-1)+'_log.csv')
    if start == 15:
        df_tmp_sample = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:
        df_tmp_sample = pd.read_csv('offline_test_label_'+str(start)+'.csv')
    
    
    user_item_ = df_tmp_log.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_dict = dict(zip(user_item_['userid'], user_item_['feedid']))
    
    user_item_candidate = df_tmp_sample.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_candidate_dict = dict(zip(user_item_candidate['userid'], user_item_candidate['feedid']))
    
    
    mm_sim_list = []
    for each_user in tqdm(df_tmp_sample['userid'].unique()) :

        if each_user not in user_item_dict:
            continue
    
        interacted_items = user_item_dict[each_user]
        candidate_items = user_item_candidate_dict[each_user]
        
        for i in candidate_items:
            sim_tmp = np.array( [ MM_model.similarity(str(i),str(j)) for j in interacted_items])
            mm_sim_list.append([each_user,i,np.mean(sim_tmp),np.sum(sim_tmp),np.min(sim_tmp),np.max(sim_tmp)])
            
    mm_sim_df = pd.DataFrame(mm_sim_list, columns=['userid', 'feedid', 'mms_mean','mms_sum','mms_min','mms_max']) 
    mms_feat = pd.merge(df_tmp_sample,mm_sim_df,how='left')
    mms_feat.to_csv('MMS_'+str(start-1)+'.csv',index=False)    
    
END_DAY = 15
start_day = 2

for start in range(start_day, END_DAY+1):
    
    df_tmp_log = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    df_tmp_log = df_tmp_log[df_tmp_log['read_comment']==1]
    if start == 15:
        df_tmp_sample = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:
        df_tmp_sample = pd.read_csv('.offline_test_label_'+str(start)+'.csv')
    
    user_item_ = df_tmp_log.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_dict = dict(zip(user_item_['userid'], user_item_['feedid']))
    
    user_item_candidate = df_tmp_sample.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_candidate_dict = dict(zip(user_item_candidate['userid'], user_item_candidate['feedid']))
    
    
    mm_sim_list = []
    for each_user in tqdm(df_tmp_sample['userid'].unique()) :

        if each_user not in user_item_dict:
            continue
    
        interacted_items = user_item_dict[each_user]
        candidate_items = user_item_candidate_dict[each_user]
        
        for i in candidate_items:
            sim_tmp = np.array( [ MM_model.similarity(str(i),str(j)) for j in interacted_items])
            mm_sim_list.append([each_user,i,np.mean(sim_tmp),np.sum(sim_tmp),np.min(sim_tmp),np.max(sim_tmp)])
            
    mm_sim_df = pd.DataFrame(mm_sim_list, columns=['userid', 'feedid', 'mms_rc_mean','mms_rc_sum','mms_rc_min','mms_rc_max']) 
    mms_feat = pd.merge(df_tmp_sample,mm_sim_df,how='left')
    mms_feat.to_csv('MMS_RC_'+str(start-1)+'.csv',index=False)    
 
for start in range(start_day, END_DAY+1):
    
    df_tmp_log = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    df_tmp_log = df_tmp_log[df_tmp_log['like']==1]
    if start == 15:
        df_tmp_sample = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:
        df_tmp_sample = pd.read_csv('offline_test_label_'+str(start)+'.csv')
    
    user_item_ = df_tmp_log.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_dict = dict(zip(user_item_['userid'], user_item_['feedid']))
    
    user_item_candidate = df_tmp_sample.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_candidate_dict = dict(zip(user_item_candidate['userid'], user_item_candidate['feedid']))
    
    
    mm_sim_list = []
    for each_user in tqdm(df_tmp_sample['userid'].unique()) :

        if each_user not in user_item_dict:
            continue
    
        interacted_items = user_item_dict[each_user]
        candidate_items = user_item_candidate_dict[each_user]
        
        for i in candidate_items:
            sim_tmp = np.array( [ MM_model.similarity(str(i),str(j)) for j in interacted_items])
            mm_sim_list.append([each_user,i,np.mean(sim_tmp),np.sum(sim_tmp),np.min(sim_tmp),np.max(sim_tmp)])
            
    mm_sim_df = pd.DataFrame(mm_sim_list, columns=['userid', 'feedid', 'mms_like_mean','mms_like_sum','mms_like_min','mms_like_max']) 
    mms_feat = pd.merge(df_tmp_sample,mm_sim_df,how='left')
    mms_feat.to_csv('MMS_Like_'+str(start-1)+'.csv',index=False)    

for start in range(start_day, END_DAY+1):
    
    df_tmp_log = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    df_tmp_log = df_tmp_log[df_tmp_log['click_avatar']==1]
    if start == 15:
        df_tmp_sample = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:
        df_tmp_sample = pd.read_csv('offline_test_label_'+str(start)+'.csv')

    
    user_item_ = df_tmp_log.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_dict = dict(zip(user_item_['userid'], user_item_['feedid']))
    
    user_item_candidate = df_tmp_sample.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_candidate_dict = dict(zip(user_item_candidate['userid'], user_item_candidate['feedid']))
    
    
    mm_sim_list = []
    for each_user in tqdm(df_tmp_sample['userid'].unique()) :

        if each_user not in user_item_dict:
            continue
    
        interacted_items = user_item_dict[each_user]
        candidate_items = user_item_candidate_dict[each_user]
        
        for i in candidate_items:
            sim_tmp = np.array( [ MM_model.similarity(str(i),str(j)) for j in interacted_items])
            mm_sim_list.append([each_user,i,np.mean(sim_tmp),np.sum(sim_tmp),np.min(sim_tmp),np.max(sim_tmp)])
            
    mm_sim_df = pd.DataFrame(mm_sim_list, columns=['userid', 'feedid', 'mms_ca_mean','mms_ca_sum','mms_ca_min','mms_ca_max']) 
    mms_feat = pd.merge(df_tmp_sample,mm_sim_df,how='left')
    mms_feat.to_csv('MMS_CA_'+str(start-1)+'.csv',index=False)    

for start in range(start_day, END_DAY+1):
    
    df_tmp_log = pd.read_csv('../../Sample_new/offline_test_'+'1_'+str(start-1)+'_log.csv')
    df_tmp_log = df_tmp_log[df_tmp_log['forward']==1]
    if start == 15:
        df_tmp_sample = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:
        df_tmp_sample = pd.read_csv('../../Sample_new/offline_test_label_'+str(start)+'.csv')
    
    user_item_ = df_tmp_log.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_dict = dict(zip(user_item_['userid'], user_item_['feedid']))
    
    user_item_candidate = df_tmp_sample.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_candidate_dict = dict(zip(user_item_candidate['userid'], user_item_candidate['feedid']))
    
    
    mm_sim_list = []
    for each_user in tqdm(df_tmp_sample['userid'].unique()) :

        if each_user not in user_item_dict:
            continue
    
        interacted_items = user_item_dict[each_user]
        candidate_items = user_item_candidate_dict[each_user]
        
        for i in candidate_items:
            sim_tmp = np.array( [ MM_model.similarity(str(i),str(j)) for j in interacted_items])
            mm_sim_list.append([each_user,i,np.mean(sim_tmp),np.sum(sim_tmp),np.min(sim_tmp),np.max(sim_tmp)])
            
    mm_sim_df = pd.DataFrame(mm_sim_list, columns=['userid', 'feedid', 'mms_forward_mean','mms_forward_sum','mms_forward_min','mms_forward_max']) 
    mms_feat = pd.merge(df_tmp_sample,mm_sim_df,how='left')
    mms_feat.to_csv('MMS_Forward_'+str(start-1)+'.csv',index=False)    


for start in range(start_day, END_DAY+1):
    
    df_tmp_log = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    df_tmp_log = df_tmp_log[df_tmp_log['follow']==1]
    if start == 15:
        df_tmp_sample = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:
        df_tmp_sample = pd.read_csv('offline_test_label_'+str(start)+'.csv')
    
    user_item_ = df_tmp_log.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_dict = dict(zip(user_item_['userid'], user_item_['feedid']))
    
    user_item_candidate = df_tmp_sample.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_candidate_dict = dict(zip(user_item_candidate['userid'], user_item_candidate['feedid']))
    
    
    mm_sim_list = []
    for each_user in tqdm(df_tmp_sample['userid'].unique()) :

        if each_user not in user_item_dict:
            continue
    
        interacted_items = user_item_dict[each_user]
        candidate_items = user_item_candidate_dict[each_user]
        
        for i in candidate_items:
            sim_tmp = np.array( [ MM_model.similarity(str(i),str(j)) for j in interacted_items])
            mm_sim_list.append([each_user,i,np.mean(sim_tmp),np.sum(sim_tmp),np.min(sim_tmp),np.max(sim_tmp)])
            
    mm_sim_df = pd.DataFrame(mm_sim_list, columns=['userid', 'feedid', 'mms_follow_mean','mms_follow_sum','mms_follow_min','mms_follow_max']) 
    mms_feat = pd.merge(df_tmp_sample,mm_sim_df,how='left')
    mms_feat.to_csv('MMS_Follow_'+str(start-1)+'.csv',index=False)    

for start in range(start_day, END_DAY+1):
    
    df_tmp_log = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    df_tmp_log = df_tmp_log[df_tmp_log['favorite']==1]
    if start == 15:
        df_tmp_sample = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:
        df_tmp_sample = pd.read_csv('offline_test_label_'+str(start)+'.csv')
    
    user_item_ = df_tmp_log.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_dict = dict(zip(user_item_['userid'], user_item_['feedid']))
    
    user_item_candidate = df_tmp_sample.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_candidate_dict = dict(zip(user_item_candidate['userid'], user_item_candidate['feedid']))
    
    
    mm_sim_list = []
    for each_user in tqdm(df_tmp_sample['userid'].unique()) :

        if each_user not in user_item_dict:
            continue
    
        interacted_items = user_item_dict[each_user]
        candidate_items = user_item_candidate_dict[each_user]
        
        for i in candidate_items:
            sim_tmp = np.array( [ MM_model.similarity(str(i),str(j)) for j in interacted_items])
            mm_sim_list.append([each_user,i,np.mean(sim_tmp),np.sum(sim_tmp),np.min(sim_tmp),np.max(sim_tmp)])
            
    mm_sim_df = pd.DataFrame(mm_sim_list, columns=['userid', 'feedid', 'mms_favorite_mean','mms_favorite_sum','mms_favorite_min','mms_favorite_max']) 
    mms_feat = pd.merge(df_tmp_sample,mm_sim_df,how='left')
    mms_feat.to_csv('MMS_Favorite_'+str(start-1)+'.csv',index=False)  
    

for start in range(start_day, END_DAY+1):
    
    df_tmp_log = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    df_tmp_log = df_tmp_log[df_tmp_log['comment']==1]
    if start == 15:
        df_tmp_sample = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:
        df_tmp_sample = pd.read_csv('offline_test_label_'+str(start)+'.csv')

    
    user_item_ = df_tmp_log.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_dict = dict(zip(user_item_['userid'], user_item_['feedid']))
    
    user_item_candidate = df_tmp_sample.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_candidate_dict = dict(zip(user_item_candidate['userid'], user_item_candidate['feedid']))
    
    
    mm_sim_list = []
    for each_user in tqdm(df_tmp_sample['userid'].unique()) :

        if each_user not in user_item_dict:
            continue
    
        interacted_items = user_item_dict[each_user]
        candidate_items = user_item_candidate_dict[each_user]
        
        for i in candidate_items:
            sim_tmp = np.array( [ MM_model.similarity(str(i),str(j)) for j in interacted_items])
            mm_sim_list.append([each_user,i,np.mean(sim_tmp),np.sum(sim_tmp),np.min(sim_tmp),np.max(sim_tmp)])
            
    mm_sim_df = pd.DataFrame(mm_sim_list, columns=['userid', 'feedid', 'mms_comment_mean','mms_comment_sum','mms_comment_min','mms_comment_max']) 
    mms_feat = pd.merge(df_tmp_sample,mm_sim_df,how='left')
    mms_feat.to_csv('MMS_Comment_'+str(start-1)+'.csv',index=False)    

for start in range(start_day, END_DAY+1):
    
    df_tmp_log = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    df_tmp_log['behavior'] = df_tmp_log[['read_comment','comment','like','click_avatar','forward','follow','favorite']].sum(axis=1)
    df_tmp_log = df_tmp_log[df_tmp_log['behavior']>0]

    if start == 15:
        df_tmp_sample = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:
        df_tmp_sample = pd.read_csv('offline_test_label_'+str(start)+'.csv')

    
    user_item_ = df_tmp_log.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_dict = dict(zip(user_item_['userid'], user_item_['feedid']))
    
    user_item_candidate = df_tmp_sample.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_candidate_dict = dict(zip(user_item_candidate['userid'], user_item_candidate['feedid']))
    
    
    mm_sim_list = []
    for each_user in tqdm(df_tmp_sample['userid'].unique()) :

        if each_user not in user_item_dict:
            continue
    
        interacted_items = user_item_dict[each_user]
        candidate_items = user_item_candidate_dict[each_user]
        
        for i in candidate_items:
            sim_tmp = np.array( [ MM_model.similarity(str(i),str(j)) for j in interacted_items])
            mm_sim_list.append([each_user,i,np.mean(sim_tmp),np.sum(sim_tmp),np.min(sim_tmp),np.max(sim_tmp)])
            
    mm_sim_df = pd.DataFrame(mm_sim_list, columns=['userid', 'feedid', 'mms_bh_mean','mms_bh_sum','mms_bh_min','mms_bh_max']) 
    mms_feat = pd.merge(df_tmp_sample,mm_sim_df,how='left')
    mms_feat.to_csv('MMS_Behavior_'+str(start-1)+'.csv',index=False)    


print('----------generate itemCF features-----------')
from collections import defaultdict
history_data = pd.read_csv('../../data/wechat/wechat_algo_data1/user_action.csv')
user_item_ = history_data.groupby('userid')['feedid'].agg(set).reset_index()
user_item_dict = dict(zip(user_item_['userid'], user_item_['feedid']))
history_data['behavior'] = history_data[['read_comment','comment','like','click_avatar','forward','follow','favorite']].sum(axis=1)
def get_sim(df_tmp):
    user_item_ = df_tmp.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_dict = dict(zip(user_item_['userid'], user_item_['feedid']))
    
    sim_item = {}  
    item_cnt = defaultdict(int)  
    for user, items in tqdm(user_item_dict.items()):  
        for i in items:  
            item_cnt[i] += 1  
            sim_item.setdefault(i, {})  
            for relate_item in items:  
                if i == relate_item:  
                    continue  
                sim_item[i].setdefault(relate_item, 0)  
                sim_item[i][relate_item] += 1 / math.log(1 + len(items)) 
                
    sim_item_corr = sim_item.copy()  
    for i, related_items in tqdm(sim_item.items()):  
        for j, cij in related_items.items():  
            sim_item_corr[i][j] = cij/math.sqrt(item_cnt[i]*item_cnt[j])
            
    return sim_item_corr

END_DAY = 15
start_day = 2

df_train = []
df_test = []

for start in range(start_day, END_DAY+1):
    
    
    history_tmp = history_data[history_data['date_']<start]
    history_tmp = history_tmp[history_tmp['read_comment']==1]
    history_tmp = history_tmp.reset_index(drop=True)
    
    
    item_sim_dict = get_sim(history_tmp)
    
    
    df_tmp_log = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    if start == 15:
        df_tmp_sample = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:
        df_tmp_sample = pd.read_csv('offline_test_label_'+str(start)+'.csv')
    
    
    user_item_ = df_tmp_log.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_dict = dict(zip(user_item_['userid'], user_item_['feedid']))
    
    
    user_item_sample = df_tmp_sample.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_sample_dict = dict(zip(user_item_sample['userid'], user_item_sample['feedid']))
    
    
    recom_item = []
    for each_user in tqdm(df_tmp_sample['userid'].unique()) :

        if each_user not in user_item_dict:
            continue
    
        interacted_items = user_item_dict[each_user]
        candidate_items = user_item_sample_dict[each_user]
        

        for j in candidate_items:
            sim_tmp = []
            for i in interacted_items:
                sim_tmp.append( item_sim_dict[i][j]) if i in item_sim_dict and j in item_sim_dict[i]  else sim_tmp.append(0)
            sim_tmp = np.array(sim_tmp)
            recom_item.append([each_user,j,np.mean(sim_tmp),np.sum(sim_tmp),np.min(sim_tmp),np.max(sim_tmp)])
 

    recom_df = pd.DataFrame(recom_item, columns=['userid', 'feedid', 'itemCF_rc_mean','itemCF_rc_sum','itemCF_rc_min','itemCF_rc_max']) 
    u2i_feat = pd.merge(df_tmp_sample,recom_df,how='left')
    u2i_feat.to_csv('User2Feed_RC_Features_'+str(start-1)+'.csv',index=False)

for start in range(start_day, END_DAY+1):
    
    
    history_tmp = history_data[history_data['date_']<start]
    history_tmp = history_tmp[history_tmp['behavior']>0]
    history_tmp = history_tmp.reset_index(drop=True)
    
    item_sim_dict = get_sim(history_tmp)
    
    df_tmp_log = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    df_tmp_log['behavior'] = df_tmp_log[['read_comment','comment','like','click_avatar','forward','follow','favorite']].sum(axis=1)
    df_tmp_log = df_tmp_log[df_tmp_log['behavior']>0]
    if start == 15:
        df_tmp_sample = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:
        df_tmp_sample = pd.read_csv('offline_test_label_'+str(start)+'.csv')
    
    user_item_ = df_tmp_log.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_dict = dict(zip(user_item_['userid'], user_item_['feedid']))
    
    
    user_item_sample = df_tmp_sample.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_sample_dict = dict(zip(user_item_sample['userid'], user_item_sample['feedid']))
    
    
    recom_item = []
    for each_user in tqdm(df_tmp_sample['userid'].unique()) :

        if each_user not in user_item_dict:
            continue
    
        interacted_items = user_item_dict[each_user]
        candidate_items = user_item_sample_dict[each_user]
        

        for j in candidate_items:
            sim_tmp = []
            for i in interacted_items:
                sim_tmp.append( item_sim_dict[i][j]) if i in item_sim_dict and j in item_sim_dict[i]  else sim_tmp.append(0)
            sim_tmp = np.array(sim_tmp)
            recom_item.append([each_user,j,np.mean(sim_tmp),np.sum(sim_tmp),np.min(sim_tmp),np.max(sim_tmp)])
 

    recom_df = pd.DataFrame(recom_item, columns=['userid', 'feedid', 'itemCF_bh_mean','itemCF_bh_sum','itemCF_bh_min','itemCF_bh_max']) 
    u2i_feat = pd.merge(df_tmp_sample,recom_df,how='left')
    u2i_feat.to_csv('User2Feed_BH_Features_'+str(start-1)+'.csv',index=False)
    

for start in range(start_day, END_DAY+1):
    
    
    history_tmp = history_data[history_data['date_']<start]
    history_tmp = history_tmp[history_tmp['like']==1]
    history_tmp = history_tmp.reset_index(drop=True)
    
    
    item_sim_dict = get_sim(history_tmp)
    
    
    df_tmp_log = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    df_tmp_log = df_tmp_log[df_tmp_log['like']==1]
    if start == 15:
        df_tmp_sample = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:
        df_tmp_sample = pd.read_csv('offline_test_label_'+str(start)+'.csv')
    
    user_item_ = df_tmp_log.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_dict = dict(zip(user_item_['userid'], user_item_['feedid']))
    
    
    user_item_sample = df_tmp_sample.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_sample_dict = dict(zip(user_item_sample['userid'], user_item_sample['feedid']))
    
    
    recom_item = []
    for each_user in tqdm(df_tmp_sample['userid'].unique()) :

        if each_user not in user_item_dict:
            continue
    
        interacted_items = user_item_dict[each_user]
        candidate_items = user_item_sample_dict[each_user]
        

        for j in candidate_items:
            sim_tmp = []
            for i in interacted_items:
                sim_tmp.append( item_sim_dict[i][j]) if i in item_sim_dict and j in item_sim_dict[i]  else sim_tmp.append(0)
            sim_tmp = np.array(sim_tmp)
            recom_item.append([each_user,j,np.mean(sim_tmp),np.sum(sim_tmp),np.min(sim_tmp),np.max(sim_tmp)])
 

    recom_df = pd.DataFrame(recom_item, columns=['userid', 'feedid', 'itemCF_like_mean','itemCF_like_sum','itemCF_like_min','itemCF_like_max']) 
    u2i_feat = pd.merge(df_tmp_sample,recom_df,how='left')
    u2i_feat.to_csv('User2Feed_Like_Features_'+str(start-1)+'.csv',index=False)
    

for start in range(start_day, END_DAY+1):
    
    
    history_tmp = history_data[history_data['date_']<start]
    history_tmp = history_tmp[history_tmp['click_avatar']==1]
    history_tmp = history_tmp.reset_index(drop=True)
    
    item_sim_dict = get_sim(history_tmp)
        
    df_tmp_log = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    df_tmp_log = df_tmp_log[df_tmp_log['click_avatar']==1]
    if start == 15:
        df_tmp_sample = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:
        df_tmp_sample = pd.read_csv('offline_test_label_'+str(start)+'.csv')
    
    
    user_item_ = df_tmp_log.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_dict = dict(zip(user_item_['userid'], user_item_['feedid']))
    
    
    user_item_sample = df_tmp_sample.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_sample_dict = dict(zip(user_item_sample['userid'], user_item_sample['feedid']))
    
    
    recom_item = []
    for each_user in tqdm(df_tmp_sample['userid'].unique()) :

        if each_user not in user_item_dict:
            continue
    
        interacted_items = user_item_dict[each_user]
        candidate_items = user_item_sample_dict[each_user]
        

        for j in candidate_items:
            sim_tmp = []
            for i in interacted_items:
                sim_tmp.append( item_sim_dict[i][j]) if i in item_sim_dict and j in item_sim_dict[i]  else sim_tmp.append(0)
            sim_tmp = np.array(sim_tmp)
            recom_item.append([each_user,j,np.mean(sim_tmp),np.sum(sim_tmp),np.min(sim_tmp),np.max(sim_tmp)])
 

    recom_df = pd.DataFrame(recom_item, columns=['userid', 'feedid', 'itemCF_ca_mean','itemCF_ca_sum','itemCF_ca_min','itemCF_ca_max']) 
    u2i_feat = pd.merge(df_tmp_sample,recom_df,how='left')
    u2i_feat.to_csv('User2Feed_CA_Features_'+str(start-1)+'.csv',index=False)
    

for start in range(start_day, END_DAY+1):        
    history_tmp = history_data[history_data['date_']<start]
    history_tmp = history_tmp[history_tmp['forward']==1]
    history_tmp = history_tmp.reset_index(drop=True)
    
    
    item_sim_dict = get_sim(history_tmp)
    
    
    df_tmp_log = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    if start == 15:
        df_tmp_sample = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:
        df_tmp_sample = pd.read_csv('offline_test_label_'+str(start)+'.csv')
    
    user_item_ = df_tmp_log.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_dict = dict(zip(user_item_['userid'], user_item_['feedid']))
    
    
    user_item_sample = df_tmp_sample.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_sample_dict = dict(zip(user_item_sample['userid'], user_item_sample['feedid']))
    
    
    recom_item = []
    for each_user in tqdm(df_tmp_sample['userid'].unique()) :

        if each_user not in user_item_dict:
            continue
    
        interacted_items = user_item_dict[each_user]
        candidate_items = user_item_sample_dict[each_user]
        

        for j in candidate_items:
            sim_tmp = []
            for i in interacted_items:
                sim_tmp.append( item_sim_dict[i][j]) if i in item_sim_dict and j in item_sim_dict[i]  else sim_tmp.append(0)
            sim_tmp = np.array(sim_tmp)
            recom_item.append([each_user,j,np.mean(sim_tmp),np.sum(sim_tmp),np.min(sim_tmp),np.max(sim_tmp)])
 

    recom_df = pd.DataFrame(recom_item, columns=['userid', 'feedid', 'itemCF_forward_mean','itemCF_forward_sum','itemCF_forward_min','itemCF_forward_max']) 
    u2i_feat = pd.merge(df_tmp_sample,recom_df,how='left')
    u2i_feat.to_csv('User2Feed_Forward_Features_'+str(start-1)+'.csv',index=False)

for start in range(start_day, END_DAY+1):
    
    
    history_tmp = history_data[history_data['date_']<start]
    history_tmp = history_tmp[history_tmp['follow']==1]
    history_tmp = history_tmp.reset_index(drop=True)
    
    
    item_sim_dict = get_sim(history_tmp)
    
    
    df_tmp_log = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    if start == 15:
        df_tmp_sample = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:
        df_tmp_sample = pd.read_csv('offline_test_label_'+str(start)+'.csv')
    
    user_item_ = df_tmp_log.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_dict = dict(zip(user_item_['userid'], user_item_['feedid']))
    
    
    user_item_sample = df_tmp_sample.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_sample_dict = dict(zip(user_item_sample['userid'], user_item_sample['feedid']))
    
    
    recom_item = []
    for each_user in tqdm(df_tmp_sample['userid'].unique()) :

        if each_user not in user_item_dict:
            continue
    
        interacted_items = user_item_dict[each_user]
        candidate_items = user_item_sample_dict[each_user]
        

        for j in candidate_items:
            sim_tmp = []
            for i in interacted_items:
                sim_tmp.append( item_sim_dict[i][j]) if i in item_sim_dict and j in item_sim_dict[i]  else sim_tmp.append(0)
            sim_tmp = np.array(sim_tmp)
            recom_item.append([each_user,j,np.mean(sim_tmp),np.sum(sim_tmp),np.min(sim_tmp),np.max(sim_tmp)])
 

    recom_df = pd.DataFrame(recom_item, columns=['userid', 'feedid', 'itemCF_follow_mean','itemCF_follow_sum','itemCF_follow_min','itemCF_follow_max']) 
    u2i_feat = pd.merge(df_tmp_sample,recom_df,how='left')
    u2i_feat.to_csv('User2Feed_Follow_Features_'+str(start-1)+'.csv',index=False)

for start in range(start_day, END_DAY+1):
    
    
    history_tmp = history_data[history_data['date_']<start]
    history_tmp = history_tmp[history_tmp['favorite']==1]
    history_tmp = history_tmp.reset_index(drop=True)
    
    
    item_sim_dict = get_sim(history_tmp)
    
    
    df_tmp_log = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    if start == 15:
        df_tmp_sample = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:
        df_tmp_sample = pd.read_csv('offline_test_label_'+str(start)+'.csv')
    
    user_item_ = df_tmp_log.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_dict = dict(zip(user_item_['userid'], user_item_['feedid']))
    
    
    user_item_sample = df_tmp_sample.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_sample_dict = dict(zip(user_item_sample['userid'], user_item_sample['feedid']))
    
    
    recom_item = []
    for each_user in tqdm(df_tmp_sample['userid'].unique()) :

        if each_user not in user_item_dict:
            continue
    
        interacted_items = user_item_dict[each_user]
        candidate_items = user_item_sample_dict[each_user]
        

        for j in candidate_items:
            sim_tmp = []
            for i in interacted_items:
                sim_tmp.append( item_sim_dict[i][j]) if i in item_sim_dict and j in item_sim_dict[i]  else sim_tmp.append(0)
            sim_tmp = np.array(sim_tmp)
            recom_item.append([each_user,j,np.mean(sim_tmp),np.sum(sim_tmp),np.min(sim_tmp),np.max(sim_tmp)])
 

    recom_df = pd.DataFrame(recom_item, columns=['userid', 'feedid', 'itemCF_favor_mean','itemCF_favor_sum','itemCF_favor_min','itemCF_favor_max']) 
    u2i_feat = pd.merge(df_tmp_sample,recom_df,how='left')
    u2i_feat.to_csv('User2Feed_Favorite_Features_'+str(start-1)+'.csv',index=False)

for start in range(start_day, END_DAY+1):
    
    
    history_tmp = history_data[history_data['date_']<start]
    history_tmp = history_tmp[history_tmp['comment']==1]
    history_tmp = history_tmp.reset_index(drop=True)
    
    
    item_sim_dict = get_sim(history_tmp)
    
    
    df_tmp_log = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    if start == 15:
        df_tmp_sample = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:
        df_tmp_sample = pd.read_csv('offline_test_label_'+str(start)+'.csv')
    
    user_item_ = df_tmp_log.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_dict = dict(zip(user_item_['userid'], user_item_['feedid']))
    
    
    user_item_sample = df_tmp_sample.groupby('userid')['feedid'].agg(set).reset_index()
    user_item_sample_dict = dict(zip(user_item_sample['userid'], user_item_sample['feedid']))
    
    
    recom_item = []
    for each_user in tqdm(df_tmp_sample['userid'].unique()) :

        if each_user not in user_item_dict:
            continue
    
        interacted_items = user_item_dict[each_user]
        candidate_items = user_item_sample_dict[each_user]
        

        for j in candidate_items:
            sim_tmp = []
            for i in interacted_items:
                sim_tmp.append( item_sim_dict[i][j]) if i in item_sim_dict and j in item_sim_dict[i]  else sim_tmp.append(0)
            sim_tmp = np.array(sim_tmp)
            recom_item.append([each_user,j,np.mean(sim_tmp),np.sum(sim_tmp),np.min(sim_tmp),np.max(sim_tmp)])
 

    recom_df = pd.DataFrame(recom_item, columns=['userid', 'feedid', 'itemCF_comment_mean','itemCF_comment_sum','itemCF_comment_min','itemCF_comment_max']) 
    u2i_feat = pd.merge(df_tmp_sample,recom_df,how='left')
    u2i_feat.to_csv('User2Feed_Comment_Features_'+str(start-1)+'.csv',index=False)
    
print('----------generate again features----------')
def is_again(x):
    if x>0:
        return 1
    else:
        return 0

def get_is_again_num(df,target_col,tag):
    df_group = df[[target_col,'is_again']].groupby(target_col,as_index=False).agg(sum)
    df_group.columns = [target_col,target_col+'_is_again'+tag]
    return df_group

END_DAY = 15
start_day = 2
target_col = 'userid'
for start in range(start_day, END_DAY+1):
    print(start)
    
    df_tmp = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    df_tmp = pd.merge(df_tmp,feedinfo,how='left')
    df_tmp['time_delta'] = df_tmp['play']*0.001-df_tmp['videoplayseconds']
    df_tmp['is_again'] = df_tmp['time_delta'].apply(lambda x: is_again(x))
    
    tag = 'new'
    sta = get_is_again_num(df_tmp,target_col,tag)
    
    sta.to_csv('User_Again_Features_'+str(start-1)+'.csv',index=False)
    
END_DAY = 15
start_day = 2
target_col = 'feedid'
for start in range(start_day, END_DAY+1):
    print(start)
    
    df_tmp = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    df_tmp = pd.merge(df_tmp,feedinfo,how='left')
    df_tmp['time_delta'] = df_tmp['play']*0.001-df_tmp['videoplayseconds']
    df_tmp['is_again'] = df_tmp['time_delta'].apply(lambda x: is_again(x))

    tag = 'new'
    sta = get_is_again_num(df_tmp,target_col,tag)
    
    sta.to_csv('Feed_Again_Features_'+str(start-1)+'.csv',index=False)

    
print('----------trend features----------')
def get_trend(df_log,df_sample,target_col,aux_col):
    df_log = df_log.rename(columns={aux_col:aux_col+'_past'})
    log_group = df_log[[target_col,aux_col+'_past']].groupby(target_col,as_index=False).agg(sum)
    
    df_sample = df_sample.rename(columns={aux_col:aux_col+'_now'})
    sample_group = df_sample[[target_col,aux_col+'_now']].groupby(target_col,as_index=False).agg(sum)
    data_merge = pd.merge(sample_group,log_group,how='left',on=[target_col])
    data_merge[target_col+'_trend'+aux_col] = data_merge[aux_col+'_now']/data_merge[aux_col+'_past']
    
    return data_merge[[target_col,target_col+'_trend'+aux_col]]

target_col = 'userid'
aux_col = 'count'
for start in range(2,16):
    df_log = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    if start==15:
        df_sample = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:
        df_sample = pd.read_csv('offline_test_label_'+str(start)+'.csv')
    
    df_log = df_log[df_log['date_']==start-1]
    df_log['count'] = 1
    df_sample['count'] = 1
    
    sta = get_trend(df_log,df_sample,target_col,aux_col)
    
    print(start)
    sta.to_csv('User_Trend_Features_'+str(start-1)+'.csv',index=False)
    
target_col = 'feedid'
aux_col = 'count'
for start in range(2,16):
    df_log = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    if start==15:
        df_sample = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:
        df_sample = pd.read_csv('offline_test_label_'+str(start)+'.csv')
    
    df_log = df_log[df_log['date_']==start-1]
    df_log['count'] = 1
    df_sample['count'] = 1
    
    sta = get_trend(df_log,df_sample,target_col,aux_col)
    
    print(start)
    sta.to_csv('Feed_Trend_Features_'+str(start-1)+'.csv',index=False)

feed_info = pd.read_csv("../../data/wechat/wechat_algo_data1/feed_info.csv")
feed_info_ = feed_info[['feedid','authorid']]
target_col = 'authorid'
aux_col = 'count'
for start in range(2,16):
    df_log = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    if start==15:
        df_sample = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:
        df_sample = pd.read_csv('offline_test_label_'+str(start)+'.csv')
    
    
    df_log = df_log[df_log['date_']==start-1]
    df_log = pd.merge(df_log,feed_info_,how='left')
    df_sample = pd.merge(df_sample,feed_info_,how='left')
    df_log['count'] = 1
    df_sample['count'] = 1
    
    sta = get_trend(df_log,df_sample,target_col,aux_col)
    
    print(start)
    sta.to_csv('Author_Trend_Features_'+str(start-1)+'.csv',index=False)

    
print('----------generate tightness features----------')
END_DAY = 15
start_day = 2
target_col = 'feedid'
auxiliary_col = 'userid'

for start in range(start_day, END_DAY+1):
    print(start)
    print('---')
    
    df_tmp_user = pd.read_csv('User_Features_'+str(start-1)+'.csv')
    df_tmp_user['play_tightness'] = np.around(df_tmp_user["userid_play_sumnew"]/df_tmp_user["feedid_countnew"],2)   
    user_sta = df_tmp_user[['userid','play_tightness']]    
    user_sta.to_csv('User_palytightness'+str(start-1)+'.csv',index=False)

END_DAY = 15
start_day = 2

for start in range(start_day, END_DAY+1):
    print(start)
    print('---')
    
    df_tmp_user = pd.read_csv('Feed_Features_'+str(start-1)+'.csv')
    df_tmp_user['feed_play_tightness'] = np.around(df_tmp_user["feedid_play_sumnew"]/df_tmp_user["userid_countnew"],2)   
    user_sta = df_tmp_user[['feedid','feed_play_tightness']]    
    user_sta.to_csv('Feed_palytightness'+str(start-1)+'.csv',index=False)
    
END_DAY = 15
start_day = 2
#before_day = 7

for start in range(start_day, END_DAY+1):
    print(start)
    print('---')
    
    df_tmp_user = pd.read_csv('Author_Features_'+str(start-1)+'.csv')
    df_tmp_user['author_feed_play_tightness'] = np.around(df_tmp_user["authorid_play_sumnew"]/df_tmp_user["authorid_feedid_numnew"],2)   
    df_tmp_user['author_user_play_tightness'] = np.around(df_tmp_user["authorid_play_sumnew"]/df_tmp_user["authorid_userid_numnew"],2)
    
    user_sta = df_tmp_user[['authorid','author_feed_play_tightness','author_user_play_tightness']]    
    user_sta.to_csv('Author_palytightness'+str(start-1)+'.csv',index=False)

    
print('----------generate entropy features----------')
def entropy(df_tmp,dm_features):
    dm_new = pd.merge(df_tmp[["userid","play"]],dm_features[["userid","userid_play_sumnew"]],how='left')
    dm_new["entropy"] = -(dm_new["play"]/dm_new["userid_play_sumnew"]) * round(np.log((dm_new["play"]+1)/dm_new["userid_play_sumnew"]),15)
    dm_new=dm_new[["userid","entropy"]].groupby("userid",as_index=False).sum()
    dm_new.columns = ["userid",'user_'+'entropy']
    return dm_new

END_DAY = 15
start_day = 2

for start in range(start_day, END_DAY+1):
    print(start)
    print('---')
    
    df_tmp = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    dm_features = pd.read_csv('User_Features_'+str(start-1)+'.csv')
    user_sta = entropy(df_tmp,dm_features)
    
    user_sta.to_csv('User_entropy_'+str(start-1)+'.csv',index=False)

def entropy(df_tmp,dm_features):
    dm_new = pd.merge(df_tmp[["feedid","play"]],dm_features[["feedid","feedid_play_sumnew"]],how='left')
    dm_new["entropy"] = -(dm_new["play"]/dm_new["feedid_play_sumnew"]) * round(np.log((dm_new["play"]+1)/dm_new["feedid_play_sumnew"]),15)
    dm_new=dm_new[["feedid","entropy"]].groupby("feedid",as_index=False).sum()
    dm_new.columns = ["feedid",'feed_'+'entropy']
    return dm_new

END_DAY = 15
start_day = 2

for start in range(start_day, END_DAY+1):
    print(start)
    print('---')
    
    df_tmp = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    dm_features = pd.read_csv('Feed_Features_'+str(start-1)+'.csv')
    user_sta = entropy(df_tmp,dm_features)
    
    user_sta.to_csv('Feed_entropy_'+str(start-1)+'.csv',index=False)
    
def entropy(df_tmp,dm_features):
    dm_new = pd.merge(df_tmp,feedinfo[['feedid','authorid']],how='left')
    dm_new = pd.merge(dm_new[["feedid","authorid","play"]],dm_features[["authorid","authorid_play_sumnew"]],how='left')
    dm_new["entropy"] = -(dm_new["play"]/dm_new["authorid_play_sumnew"]) * round(np.log((dm_new["play"]+1)/dm_new["authorid_play_sumnew"]),15)
    dm_new=dm_new[["authorid","entropy"]].groupby("authorid",as_index=False).sum()
    dm_new.columns = ["authorid",'author_'+'entropy']
    return dm_new

END_DAY = 15
start_day = 2

for start in range(start_day, END_DAY+1):
    print(start)
    print('---')
    
    df_tmp = pd.read_csv('offline_test_'+'1_'+str(start-1)+'_log.csv')
    dm_features = pd.read_csv('Author_Features_'+str(start-1)+'.csv')
    user_sta = entropy(df_tmp,dm_features)
    
    user_sta.to_csv('Author_entropy_'+str(start-1)+'.csv',index=False)


    
print('----------generate TodayRank features----------')
for start in range(2,16):
    if start==15:
        df_tmp = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:
        df_tmp = pd.read_csv('offline_test_label_'+str(start)+'.csv')
    df_tmp['feed_count_today'] = 1
    df_group = df_tmp[['feedid','feed_count_today']].groupby('feedid',as_index=False).agg(sum)
    df_group['feed_count_rank'] = df_group[['feed_count_today']].rank()
    df_group['feed_count_percentage'] = df_group[['feed_count_today']].rank(pct=True)
    df_group.to_csv('Today_FeedRank_'+str(start)+'.csv',index=False)
    print(start)
    
for start in range(2,16):
    if start==15:
        df_tmp = pd.read_csv('../../data/wechat/wechat_algo_data1_b/test_b.csv')
    else:
        df_tmp = pd.read_csv('offline_test_label_'+str(start)+'.csv')
    df_tmp = pd.merge(df_tmp,feedinfo[['feedid','authorid']],how='left')
    df_tmp['author_count_today'] = 1
    df_group = df_tmp[['authorid','author_count_today']].groupby('authorid',as_index=False).agg(sum)
    df_group['author_count_rank'] = df_group[['author_count_today']].rank()
    df_group['author_count_percentage'] = df_group[['author_count_today']].rank(pct=True)
    df_group.to_csv('Today_AuthorRank_'+str(start)+'.csv',index=False)
    print(start)


print('-----------generate UserDate features----------')
pro_col = ['userid_read_comment_sumnew', 'userid_comment_sumnew',
       'userid_like_sumnew', 'userid_play_sumnew', 'userid_stay_sumnew',
       'userid_click_avatar_sumnew', 'userid_forward_sumnew',
       'userid_follow_sumnew', 'userid_favorite_sumnew',
       'userid_behavior_sumnew', 'userid_behavior_flag_sumnew']  
for i in range(1,15):
    df = pd.read_csv('User_Features_'+str(i)+'.csv')
    df_tmp = df[['userid','userid_date__numnew']+pro_col]
    
    for each_feat in pro_col:
        hyper = HyperParam(1, 1)
        hyper.update_from_data_by_moment(df_tmp['userid_date__numnew'], df_tmp[each_feat])
        df_tmp[each_feat+'_daterate'] = ctr(df_tmp['userid_date__numnew'], df_tmp[each_feat],hyper.alpha,hyper.beta)
    
    df_tmp[['userid']+[x+'_daterate' for x in pro_col]].to_csv('User_Date_Features_'+str(i)+'.csv',index=False)
    print(i)
    
print('----------generate user_5days features----------')
def getNumFeatures(df_his,target_col,auxiliary_col,col_list, suffix):
    tmp_sta = df_his.groupby(target_col,as_index=False)[col_list].agg(set)
    col_list_new = [target_col + '_' + x+'_set' for x in col_list]
    tmp_sta.columns = [target_col] + col_list_new
    for each_feat in col_list_new:
        tmp_sta[ each_feat[:-4]+'_num'+suffix ] = tmp_sta[each_feat].apply(lambda x:len(x))
    
    count_sta = df_his[[target_col,auxiliary_col]].groupby([target_col],as_index=False).count()
    count_sta = count_sta.rename(columns={auxiliary_col:auxiliary_col+'_count'+suffix})
    tmp_sta = pd.merge(tmp_sta,count_sta,how='left',on=[target_col])
    
    return tmp_sta[[target_col,auxiliary_col+'_count'+suffix]+[x for x in tmp_sta.columns if '_num' in x]]

def getSumFeatures(df_his, target_col, col_list, df_tmp, suffix):
    tmp_sta = df_his.groupby(target_col,as_index=False)[col_list].sum()
    col_list_new = [target_col + '_' + x+'_sum'+suffix for x in col_list]
    tmp_sta = tmp_sta.rename(columns=dict(zip(col_list,col_list_new)))
    return tmp_sta
def getFeature(df, target_col,auxiliary_col,col_list_num,col_list_sum,tag):
    user_sta_1 = getNumFeatures(df,target_col,auxiliary_col,col_list_num,tag)
    user_sta_2 = getSumFeatures(df, target_col, col_list_sum, user_sta_1[auxiliary_col+'_count'+tag],suffix=tag)
    user_sta = pd.merge(user_sta_1,user_sta_2,how='left')
    
    return user_sta 

END_DAY = 15
start_day = 2
n_day = 5

target_col = 'userid'
auxiliary_col = 'feedid'
col_list_num = ['feedid','authorid','date_','device']
col_list_sum = ['read_comment','comment','like','play','stay','click_avatar','forward','follow','favorite','behavior','behavior_flag']
df_his = pd.read_csv('../../data/wechat/wechat_algo_data1/user_action.csv')
for start in range(start_day, END_DAY+1):
    print(start)
    print('---')
    
    left, right = max(start_day - n_day, 1), start_day - 1
    
    df_tmp = df_his[ (df_his['date_']<=right) & (df_his['date_']>=right)].reset_index(drop=True)
    
    df_tmp = pd.merge(df_tmp,feedinfo[['feedid','authorid']],how='left')
    df_tmp['behavior'] = df_tmp [['read_comment','comment','like','click_avatar','forward','follow','favorite']].sum(axis=1)
    df_tmp['behavior_flag'] = df_tmp['behavior'].apply(lambda x: 1 if x>=1 else 0)
        
    tag = '5day'
    user_sta = getFeature(df_tmp, target_col,auxiliary_col,col_list_num,col_list_sum,tag)
    
    user_sta.to_csv('User_Features_5day_'+str(start-1)+'.csv',index=False)

print('----------generate feed_5days features----------')
def getNumFeatures(df_his,target_col,auxiliary_col,col_list, suffix):
    tmp_sta = df_his.groupby(target_col,as_index=False)[col_list].agg(set)
    col_list_new = [target_col + '_' + x+'_set' for x in col_list]
    tmp_sta.columns = [target_col] + col_list_new
    for each_feat in col_list_new:
        tmp_sta[ each_feat[:-4]+'_num'+suffix ] = tmp_sta[each_feat].apply(lambda x:len(x))
    
    count_sta = df_his[[target_col,auxiliary_col]].groupby([target_col],as_index=False).count()
    count_sta = count_sta.rename(columns={auxiliary_col:auxiliary_col+'_count'+suffix})
    tmp_sta = pd.merge(tmp_sta,count_sta,how='left',on=[target_col])
    
    return tmp_sta[[target_col,auxiliary_col+'_count'+suffix]+[x for x in tmp_sta.columns if '_num' in x]]

def getSumFeatures(df_his, target_col, col_list, df_tmp, suffix):
    tmp_sta = df_his.groupby(target_col,as_index=False)[col_list].sum()
    col_list_new = [target_col + '_' + x+'_sum'+suffix for x in col_list]
    tmp_sta = tmp_sta.rename(columns=dict(zip(col_list,col_list_new)))
    return tmp_sta

def getFeature(df, target_col,auxiliary_col,col_list_num,col_list_sum,tag):
    user_sta_1 = getNumFeatures(df,target_col,auxiliary_col,col_list_num,suffix=tag)
    user_sta_2 = getSumFeatures(df, target_col, col_list_sum, user_sta_1[auxiliary_col+'_count'+tag],suffix=tag)
    user_sta = pd.merge(user_sta_1,user_sta_2,how='left')
    
    return user_sta 

END_DAY = 15
start_day = 2
n_day = 5

target_col = 'feedid'
auxiliary_col = 'userid'
col_list_sum = ['read_comment','comment','like','play','stay','click_avatar','forward','follow','favorite','behavior','behavior_flag']
col_list_num = ['userid','date_','device']

for start in range(start_day, END_DAY+1):
    print(start)
    print('---')
    
    left, right = max(start_day - n_day, 1), start_day - 1
    
    df_tmp = df_his[ (df_his['date_']<=right) & (df_his['date_']>=right)].reset_index(drop=True)
    
    df_tmp['behavior'] = df_tmp [['read_comment','comment','like','click_avatar','forward','follow','favorite']].sum(axis=1)
    df_tmp['behavior_flag'] = df_tmp['behavior'].apply(lambda x: 1 if x>=1 else 0)
        
    tag = '5day'
    feed_sta = getFeature(df_tmp, target_col,auxiliary_col,col_list_num,col_list_sum,tag)
        
    feed_sta.to_csv('Feed_Features_5day_'+str(start-1)+'.csv',index=False)
    
print('----------generate author_5days features----------')
def getNumFeatures(df_his,target_col,col_list, suffix):
    tmp_sta = df_his.groupby(target_col,as_index=False)[col_list].agg(set)
    col_list_new = [target_col + '_' + x+'_set' for x in col_list]
    tmp_sta = tmp_sta.rename(columns=dict(zip(col_list,col_list_new)))
    for each_feat in col_list_new:
        tmp_sta[ each_feat[:-4]+'_num'+suffix ] = tmp_sta[each_feat].apply(lambda x:len(x))
    
    return tmp_sta[[target_col]+[x for x in tmp_sta.columns if '_num' in x]]

def getSumFeatures(df_his, target_col, col_list, df_tmp, suffix):
    df_his['authorid_count5day'] = 1
    tmp_sta = df_his.groupby(target_col,as_index=False)[col_list+['authorid_count5day']].sum()
    col_list_new = [target_col + '_' + x+'_sum'+suffix for x in col_list]
    tmp_sta = tmp_sta.rename(columns=dict(zip(col_list,col_list_new)))
    return tmp_sta

def getFeature(df,target_col,col_list_num,col_list_sum,tag):
    sta_1 = getNumFeatures(df,target_col,col_list_num,tag)

    sta_2 = getSumFeatures(df, target_col, col_list_sum, sta_1['authorid_feedid_num'+tag],suffix=tag)
    sta = pd.merge(sta_1,sta_2,how='left')
    
    return sta 

END_DAY = 15
start_day = 2
n_day = 5

target_col = 'authorid'
col_list_num = ['userid','feedid','date_','device']
col_list_sum = ['read_comment','comment','like','play','stay','click_avatar','forward','follow','favorite']

for start in range(start_day, END_DAY+1):
    print(start)
    print('---')
    
    left, right = max(start_day - n_day, 1), start_day - 1
    
    df_tmp = df_his[ (df_his['date_']<=right) & (df_his['date_']>=right)].reset_index(drop=True)
    
    df_tmp = pd.merge(df_tmp,feedinfo[['feedid','authorid']],how='left')
    
    tag = '5day'
    author_sta = getFeature(df_tmp, target_col,col_list_num,col_list_sum,tag)
    
    author_sta.to_csv('Author_Features_5day_'+str(start-1)+'.csv',index=False)
    
print('----------generate user_author_5days features----------')
END_DAY = 15
start_day = 2
n_day = 5

for start in range(start_day, END_DAY+1):
    print(start)
    print('---')
    
    left, right = max(start_day - n_day, 1), start_day - 1
    
    df_tmp = df_his[ (df_his['date_']<=right) & (df_his['date_']>=right)].reset_index(drop=True)
    
    
    
    df_tmp = pd.merge(df_tmp,feedinfo[['feedid','authorid']],how='left')
    
    user_author_sta = df_tmp.groupby(['userid','authorid'],as_index=False)['read_comment','comment','like','play',
                                                           'stay','click_avatar','forward','follow','favorite'].sum()
    
    col = [x for x in user_author_sta if x not in ['userid','authorid']]
    user_author_sta = user_author_sta.rename(columns=dict(zip(col,['ua_5day'+x for x in col])))
    
    user_author_sta.to_csv('User_Author_Interaction_Features_5day_'+str(start-1)+'.csv',index=False)
    

print('----------generate embedding_PCA features----------')
embedding = pd.read_csv("../../data/wechat/wechat_algo_data1/feed_embeddings.csv")
embedding['feed_embedding'] = embedding['feed_embedding'].apply(lambda x: x.split(' '))
for i in range(512):
    embedding['emfeat_'+str(i)] = embedding['feed_embedding'].apply(lambda x: x[i])
from sklearn import decomposition
pca = decomposition.PCA(n_components=32)
pca_feats = ['emfeat_'+str(x) for x in range(512)]
pca.fit(embedding[pca_feats])
embed_new = pca.transform(embedding[pca_feats])
embed_feature = pd.DataFrame(embed_new)
embed_feature.columns = ['emfeat_'+str(x) for x in range(32)]
embed_feature['feedid'] = embedding['feedid']
embed_feature.to_csv('Embedding_PCA_Features.csv',index=False)

print('----------generate UserKey features----------')
def deal_key(x):
    if type(x)==str:
        return x.split(';')
    else:
        return []
feedinfo['key_list'] = feedinfo['manual_keyword_list'].apply(lambda x: deal_key(x))
for start in range(2,16):
    if start == 15:
        df_tmp_sample = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:
        df_tmp_sample = pd.read_csv('.offline_test_label_'+str(start)+'.csv')
    df = pd.merge(df_log,feedinfo[['feedid','key_list']],how='left')
    df = df.explode('key_list')
    df['uk_count'] = 1
    pro_col = ['read_comment','like','forward','click_avatar','follow','favorite','comment']
    df_group = df[['userid','tag_list','read_comment','like','forward','click_avatar','follow','favorite','comment','uk_count']].groupby(['userid','key_list'],as_index=False).agg(sum)
    for each_feat in pro_col:
        hyper = HyperParam(1, 1)
        hyper.update_from_data_by_moment(df_group['uk_count'], df_group[each_feat])
        df_group[each_feat+'_pro'] = ctr(df_group['uk_count'], df_group[each_feat],hyper.alpha,hyper.beta)

    pro_feats = [x+'_pro' for x in pro_col]
    df_sample = pd.merge(df_sample,feedinfo[['feedid','key_list']],how='left')
    df_sample = df_sample.explode('key_list')
    df_merge = pd.merge(df_sample[['userid','feedid','key_list']],df_group[['userid','key_list']+pro_feats],how = 'left',on=['userid','key_list'])
    sta = df_merge.groupby(['userid','feedid'],as_index=False).agg(sum)
    sta.to_csv('UserKey_sta_'+str(start)+'.csv',index=False)
    print(start)

print('----------generate UserTag features----------')
def deal_tag(x):
    if type(x)==str:
        return x.split(';')
    else:
        return []
feedinfo = pd.read_csv("../../data/wechat/wechat_algo_data1/feed_info.csv")
user_action = pd.read_csv("../../data/wechat/wechat_algo_data1/user_action.csv")
feedinfo['tag_list'] = feedinfo['manual_tag_list'].apply(lambda x: deal_tag(x))
data = pd.merge(user_action,feedinfo[['feedid','authorid','tag_list']],how='left')
data = data.explode('tag_list')
behavior_group = data[['tag_list','read_comment','like','forward','click_avatar','follow','favorite','comment']].groupby('tag_list',as_index=False).agg(sum)
behavior_group_ = data[['tag_list','read_comment','like','forward','click_avatar','follow','favorite','comment']].groupby('tag_list',as_index=False).agg(np.mean)
behavior_group_.columns = ['tag_list','tag_read_comment','tag_like','tag_forward','tag_click_avatar','tag_follow','tag_favorite','tag_comment']
for start in range(2,16):
    df_log = pd.read_csv('../Generate_Sample/offline_test_1_'+str(start-1)+'_log.csv')
    if start==15:
        df_sample = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:
        df_sample = pd.read_csv('offline_test_label_'+str(start)+'.csv')
        
    df = pd.merge(df_log,feedinfo[['feedid','tag_list']],how='left')
    df = df.explode('tag_list')
    df['ut_count'] = 1
    pro_col = ['read_comment','like','forward','click_avatar','follow','favorite','comment']
    df_group = df[['userid','tag_list','read_comment','like','forward','click_avatar','follow','favorite','comment','ut_count']].groupby(['userid','tag_list'],as_index=False).agg(sum)
    for each_feat in pro_col:
        hyper = HyperParam(1, 1)
        hyper.update_from_data_by_moment(df_group['ut_count'], df_group[each_feat])
        df_group[each_feat+'_pro'] = ctr(df_group['ut_count'], df_group[each_feat],hyper.alpha,hyper.beta)

    pro_feats = [x+'_pro' for x in pro_col]
    df_sample = pd.merge(df_sample,feedinfo[['feedid','tag_list']],how='left')
    df_sample = df_sample.explode('tag_list')
    df_merge = pd.merge(df_sample[['userid','feedid','tag_list']],df_group[['userid','tag_list']+pro_feats],how = 'left',on=['userid','tag_list'])
    sta = df_merge.groupby(['userid','feedid'],as_index=False).agg(sum)
    sta.to_csv('UserTag_sta_'+str(start)+'.csv',index=False)
    print(start)
    
print('-----------generate tfidf_sum features----------')
df = pd.read_csv('../../data/wechat/wechat_algo_data1/feed_info.csv')
feedlist = []
manual_tag_list = []

for i, row in df.iterrows():
    tmp = row['manual_tag_list']
    try :
        array = tmp.split(';')
        array = [int(x) for x in array]
        array.sort()
        feedlist.append(row['feedid'])
        manual_tag_list.append(array)
    except:
        print('nan')
feed2mtag = dict(zip(feedlist,manual_tag_list))
df['manual_tag_list'] = df['feedid'].apply(lambda x: feed2mtag[x] if x in feed2mtag else [-1])
tag_count = {}
for i,row in df.iterrows():
    tmp = row['manual_tag_list']
    for each_tag in tmp:
        if each_tag not in tag_count:
            tag_count[each_tag] = 1
        else:
            tag_count[each_tag] += 1
voc_dict = {}
df_dict = {}

count = 0
for i,row in df.iterrows():
    for each_word in row['manual_tag_list']:
        if each_word not in voc_dict:
            voc_dict[each_word] = count
            count += 1
        if each_word not in df_dict:
            df_dict[each_word] = 1
        else:
            df_dict[each_word] += 1
idf_dict = {}
for each_word in df_dict:
    idf_dict[each_word] = df.shape[0]/df_dict[each_word]
tfidf = {}
for i,row in df.iterrows():
    tmp = np.zeros(len(voc_dict))
    for each_word in row['manual_tag_list']:
        tmp[voc_dict[each_word]] = np.log(df.shape[0]/df_dict[each_word])
    
    tfidf[row['feedid']] = tmp
tmp = feedinfo[['feedid']]
tmp['tfidf_sum'] = tmp['feedid'].apply(lambda x: tfidf[x])
tmp.to_csv('tdidf_sum.csv',index=False)


print('----------generate FeedTopn features----------')
from gensim.models import KeyedVectors
MM_model = KeyedVectors.load_word2vec_format('feed_embeddings_new.csv')
feedinfo = pd.read_csv('../../data/wechat/wechat_algo_data1/feed_embeddings.csv')
feedlist = feedinfo['feedid']
feed2topn = dict()

for each_feed in tqdm(feedlist):
    
    feed_topn = MM_model.most_similar(str(each_feed),topn=10)
    feed_topn = [ int(x[0]) for x in feed_topn]
    
    feed2topn[each_feed] = feed_topn

bh_list = ['read_comment','comment','like','click_avatar','forward','follow','favorite','behavior_flag']
def get_simpro(df_tmp,x,behave):
    df_new = df_tmp[df_tmp['feedid'].isin(x)]
    return np.mean(df_new['feedid_'+behave+'_sumnew_pro'])

for start in range(2,16):
    df_tmp = pd.read_csv('Feed_Pro_Features_'+str(start-1)+'.csv')
    if start==15:
        df_test = pd.read_csv("../../data/wechat/wechat_algo_data1_b/test_b.csv")
    else:
        df_test = pd.read_csv("offline_test_label_"+str(start)+'.csv')
    df_test = df_test[['feedid']].drop_duplicates().reset_index(drop=True)
    df_test['feedid_topn'] = df_test['feedid'].apply(lambda x: feed2topn[x])
    
    for i in tqdm(bh_list):
        df_test['feed_topn_'+i+'_pro'] = df_test['feedid_topn'].apply(lambda x: get_simpro(df_tmp,x,i))
    
    col = ['feed_topn_'+x+'_pro' for x in bh_list]
    print(start)
    df_test[['feedid']+col].to_csv('Feed_Topn_'+str(start)+'.csv',index=False)

    
print("----------generate word features----------")
df = pd.read_csv("../../data/wechat/wechat_algo_data1/feed_info.csv")
feedlist = []
manual_tag_list = []

for i, row in df.iterrows():
    tmp = row['manual_tag_list']
    if type(tmp)==str:
        array = tmp.split(';')
        array = [int(x) for x in array]
        array.sort()
        feedlist.append(row['feedid'])
        manual_tag_list.append(array)

feed2mtag = dict(zip(feedlist,manual_tag_list))
df['manual_tag_list'] = df['feedid'].apply(lambda x: feed2mtag[x] if x in feed2mtag else [-1])
tag_count = {}
for i,row in df.iterrows():
    tmp = row['manual_tag_list']
    for each_tag in tmp:
        if each_tag not in tag_count:
            tag_count[each_tag] = 1
        else:
            tag_count[each_tag] += 1
df['manual_tag_list_str'] = df['manual_tag_list'].apply(lambda x: ','.join([str(k) for k in x]))
voc_dict = {}
df_dict = {}

count = 0
for i,row in df.iterrows():
    for each_word in row['manual_tag_list']:
        if each_word not in voc_dict:
            voc_dict[each_word] = count
            count += 1
        if each_word not in df_dict:
            df_dict[each_word] = 1
        else:
            df_dict[each_word] += 1
idf_dict = {}
for each_word in df_dict:
    idf_dict[each_word] = df.shape[0]/df_dict[each_word]

ls_word = []
for i in df_dict:
    if df_dict[i]>=100:
        ls_word.append(i)

df_ = df[['feedid','authorid','manual_tag_list']]
def get_dummy(x,word):
    if word in x:
        return 1
    else:
        return 0

for word in ls_word:
    df_['manual_tag'+str(word)] = df_['manual_tag_list'].apply(lambda x: get_dummy(x,word))

user_action = pd.read_csv("../../data/wechat/wechat_algo_data1/user_action.csv")
data = pd.merge(user_action,df_,how='left')
df_train = data[data['date_']<=13]
df_test = data[data['date_']==14]
feats = ['manual_tag'+str(x) for x in ls_word]
import lightgbm as lgb
clf = lgb.LGBMClassifier(learning_rate=0.01,max_depth=6,n_estimators=2000,random_state=2020,num_leaves=64,
                         n_jobs=22,subsample=0.8,subsample_freq=5,colsample_bytree=0.8)
sample_0 = df_train[df_train['read_comment']==0].sample(len(df_train[df_train['read_comment']==1])*10,random_state=2020)
sample = sample_0.append(df_train[df_train['read_comment']==1])
clf.fit(sample[feats],sample['read_comment'])
y_pred = clf.predict_proba(df_test[feats])[:,1]
df_test['read_comment_prob'] = y_pred

booster = clf.booster_
importance = booster.feature_importance(importance_type='split')
feature_name = booster.feature_name()

feature_importance = pd.DataFrame({'feature_name':feature_name,'importance':importance} )
feat_new = feature_importance[feature_importance['importance']>=300]
feats = list(feat_new['feature_name'])
word_feature = df_[['feedid']+feats]
word_feature.to_csv('word_feature.csv',index=False)


    






