# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 00:08:44 2021

@author: hp
"""


import pandas as pd
import numpy as np

feedinfo = pd.read_csv("../../data/wechat/wechat_algo_data1/feed_info.csv")

df_train = []
df_test = []

for start in range(3,16):
    
    if start == 15:
        df_tmp = pd.read_csv('../../data/wechat/wechat_algo_data1/test_b.csv')
    else:    
        df_tmp = pd.read_csv('offline_test_label_'+str(start)+'.csv')
    
    user_tmp = pd.read_csv('User_Features_'+str(start-1)+'.csv')
    feed_tmp = pd.read_csv('Feed_Features_'+str(start-1)+'.csv')
    author_tmp = pd.read_csv('Author_Features_'+str(start-1)+'.csv')
    ua_interaction_tmp = pd.read_csv('User_Author_Interaction_Features_'+str(start-1)+'.csv')
    del ua_interaction_tmp['ua_count']
    mms_tmp = pd.read_csv('MMS_'+str(start-1)+'.csv')
    mms_rc = pd.read_csv('MMS_RC_'+str(start-1)+'.csv')
    mms_like = pd.read_csv('MMS_Like_'+str(start-1)+'.csv')
    mms_ca = pd.read_csv('MMS_CA_'+str(start-1)+'.csv')
    mms_forward = pd.read_csv('MMS_Forward_'+str(start-1)+'.csv')
    mms_favorite = pd.read_csv('MMS_Favorite_'+str(start-1)+'.csv')
    mms_follow = pd.read_csv('MMS_Follow_'+str(start-1)+'.csv')
    mms_comment = pd.read_csv('MMS_Comment_'+str(start-1)+'.csv')
    mms_bh = pd.read_csv('MMS_Behavior_'+str(start-1)+'.csv')
    user_pro_tmp = pd.read_csv('User_Pro_Features_'+str(start-1)+'.csv')
    del user_pro_tmp['Unnamed: 0']
    feed_pro_tmp = pd.read_csv('Feed_Pro_Features_'+str(start-1)+'.csv')
    del feed_pro_tmp['Unnamed: 0']
    author_pro_tmp = pd.read_csv('Author_Pro_Features_'+str(start-1)+'.csv')
    user_again = pd.read_csv('User_Again_Features_'+str(start-1)+'.csv')
    feed_again = pd.read_csv('Feed_Again_Features_'+str(start-1)+'.csv')
     
    u2f_rc_feat_tmp = pd.read_csv('User2Feed_RC_Features_'+str(start-1)+'.csv')
    u2f_bh_feat_tmp = pd.read_csv('User2Feed_BH_Features_'+str(start-1)+'.csv')
    u2f_like_feat_tmp = pd.read_csv('User2Feed_Like_Features_'+str(start-1)+'.csv')
    u2f_ca_feat_tmp = pd.read_csv('User2Feed_CA_Features_'+str(start-1)+'.csv')
    u2f_forward_feat_tmp = pd.read_csv('User2Feed_Forward_Features_'+str(start-1)+'.csv')
    u2f_follow_feat_tmp = pd.read_csv('User2Feed_Follow_Features_'+str(start-1)+'.csv')
    u2f_favorite_feat_tmp = pd.read_csv('User2Feed_Favorite_Features_'+str(start-1)+'.csv')
    u2f_comment_feat_tmp = pd.read_csv('User2Feed_Comment_Features_'+str(start-1)+'.csv')
    
    
    feed_trend = pd.read_csv('Feed_Trend_Features_'+str(start-1)+'.csv')
    user_trend = pd.read_csv('User_Trend_Features_'+str(start-1)+'.csv')
    author_trend = pd.read_csv('Author_Trend_Features_'+str(start-1)+'.csv')
    
    
    user_tight = pd.read_csv('User_palytightness'+str(start-1)+'.csv')
    feed_tight = pd.read_csv('Feed_palytightness'+str(start-1)+'.csv')
    author_tight = pd.read_csv('Author_palytightness'+str(start-1)+'.csv')

    
    user_entropy = pd.read_csv('User_entropy_'+str(start-1)+'.csv')
    feed_entropy = pd.read_csv('Feed_entropy_'+str(start-1)+'.csv')
    author_entropy = pd.read_csv('Author_entropy_'+str(start-1)+'.csv')
    
    
    feed_rank_today = pd.read_csv('Today_FeedRank_'+str(start)+'.csv')
    author_rank_today = pd.read_csv('Today_AuthorRank_'+str(start)+'.csv')
    
    user_date = pd.read_csv('User_Date_Features_'+str(start-1)+'.csv')
    del user_date['Unnamed: 0']
    tag_sta = pd.read_csv('UserTag_sta_'+str(start)+'.csv')
    uk = pd.read_csv('UserKey_sta_'+str(start)+'.csv')
    feed_topn = pd.read_csv('Feed_Topn_'+str(start)+'.csv')
    
    print(start)
    
    df_tmp = pd.merge(df_tmp,feedinfo[['feedid','authorid','videoplayseconds','bgm_song_id','bgm_singer_id']])
    df_tmp = pd.merge(df_tmp,user_tmp,how='left')
    df_tmp = pd.merge(df_tmp,feed_tmp,how='left')
    df_tmp = pd.merge(df_tmp,author_tmp,how='left')
    df_tmp = pd.merge(df_tmp,user_pro_tmp,how='left')
    df_tmp = pd.merge(df_tmp,feed_pro_tmp,how='left')
    df_tmp = pd.merge(df_tmp,author_pro_tmp,how='left')
    df_tmp = pd.merge(df_tmp,user_again,how='left')
    df_tmp = pd.merge(df_tmp,feed_again,how='left')
    df_tmp = pd.merge(df_tmp,u2f_rc_feat_tmp,how='left')
    df_tmp = pd.merge(df_tmp,u2f_bh_feat_tmp,how='left')
    df_tmp = pd.merge(df_tmp,u2f_like_feat_tmp,how='left')
    df_tmp = pd.merge(df_tmp,u2f_ca_feat_tmp,how='left')
    df_tmp = pd.merge(df_tmp,u2f_forward_feat_tmp,how='left')
    df_tmp = pd.merge(df_tmp,u2f_follow_feat_tmp,how='left')
    df_tmp = pd.merge(df_tmp,u2f_favorite_feat_tmp,how='left')
    df_tmp = pd.merge(df_tmp,u2f_comment_feat_tmp,how='left')   
    df_tmp = pd.merge(df_tmp,feed_trend,how='left')
    df_tmp = pd.merge(df_tmp,user_trend,how='left')
    df_tmp = pd.merge(df_tmp,author_trend,how='left')
    df_tmp = pd.merge(df_tmp,user_tight,how='left')
    df_tmp = pd.merge(df_tmp,feed_tight,how='left')
    df_tmp = pd.merge(df_tmp,author_tight,how='left')
    df_tmp = pd.merge(df_tmp,user_entropy,how='left')
    df_tmp = pd.merge(df_tmp,feed_entropy,how='left')
    df_tmp = pd.merge(df_tmp,author_entropy,how='left')
    df_tmp = pd.merge(df_tmp,feed_rank_today,how='left')
    df_tmp = pd.merge(df_tmp,author_rank_today,how='left')
    df_tmp = pd.merge(df_tmp,user_date,how='left')
    df_tmp = pd.merge(df_tmp,tag_sta,how='left')
    df_tmp = pd.merge(df_tmp,uk,how='left')
    df_tmp = pd.merge(df_tmp,feed_topn,how='left')
    
    
    
    df_tmp['ua_id'] = df_tmp['userid'].astype(str)+','+df_tmp['authorid'].astype(str)
    ua_interaction_tmp['ua_id'] = ua_interaction_tmp['userid'].astype(str)+','+ua_interaction_tmp['authorid'].astype(str)
    del ua_interaction_tmp['userid']
    del ua_interaction_tmp['authorid']
    df_tmp = pd.merge(df_tmp,ua_interaction_tmp,how='left',on=['ua_id'])    
    
    df_tmp = pd.merge(df_tmp,mms_tmp[['userid','feedid','mms_mean','mms_sum','mms_min','mms_max']],how='left')
    df_tmp = pd.merge(df_tmp,mms_rc[['userid','feedid','mms_rc_mean','mms_rc_sum','mms_rc_min','mms_rc_max']],how='left')
    df_tmp = pd.merge(df_tmp,mms_like[['userid','feedid','mms_like_mean','mms_like_sum','mms_like_min','mms_like_max']],how='left')
    df_tmp = pd.merge(df_tmp,mms_ca[['userid','feedid','mms_ca_mean','mms_ca_sum','mms_ca_min','mms_ca_max']],how='left')
    df_tmp = pd.merge(df_tmp,mms_forward[['userid','feedid','mms_forward_mean','mms_forward_sum','mms_forward_min','mms_forward_max']],how='left')
    df_tmp = pd.merge(df_tmp,mms_follow[['userid','feedid','mms_follow_mean','mms_follow_sum','mms_follow_min','mms_follow_max']],how='left')
    df_tmp = pd.merge(df_tmp,mms_favorite[['userid','feedid','mms_favorite_mean','mms_favorite_sum','mms_favorite_min','mms_favorite_max']],how='left')
    df_tmp = pd.merge(df_tmp,mms_comment[['userid','feedid','mms_comment_mean','mms_comment_sum','mms_comment_min','mms_comment_max']],how='left')
    df_tmp = pd.merge(df_tmp,mms_bh[['userid','feedid','mms_bh_mean','mms_bh_sum','mms_bh_min','mms_bh_max']],how='left')
    if start == 15:
        df_test = df_tmp
        break
    
    if len(df_train) == 0:
        df_train = df_tmp
    else:
        df_train = pd.concat([df_train,df_tmp])

statistic_feature = pd.read_csv("word_statistic_features.csv")
df_train = pd.merge(df_train,statistic_feature,how='left',on=['authorid'])
df_test = pd.merge(df_test,statistic_feature,how='left',on=['authorid'])

tag_tmp = pd.read_csv("word_feature.csv")
df_train = pd.merge(df_train,tag_tmp,how='left')
df_test = pd.merge(df_test,tag_tmp,how='left')

keyword_tmp = pd.read_csv("manual_keyword_words_new.csv") 
df_train = pd.merge(df_train,keyword_tmp,how='left')
df_test = pd.merge(df_test,keyword_tmp,how='left')

tdidf_sum = pd.read_csv("tdidf_sum.csv")
df_train = pd.merge(df_train,tdidf_sum,how='left')
df_test = pd.merge(df_test,tdidf_sum,how='left')

keyword_sum = pd.read_csv("keyword_tdidf_sum.csv")
df_train = pd.merge(df_train,keyword_sum,how='left')
df_test = pd.merge(df_test,keyword_sum,how='left')

embed_pca = pd.read_csv("Embedding_PCA_Features.csv")
df_train = pd.merge(df_train,embed_pca,how='left')
df_test = pd.merge(df_test,embed_pca,how='left')

df_train.to_csv('prepare/train_for_tree.csv',index=False)
df_test.to_csv('prepare/test_for_tree.csv',index=False)