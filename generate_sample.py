# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 11:04:36 2021

@author: hp
"""


import numpy as np
import pandas as pd

history_data = pd.read_csv('../../data/wechat/wechat_algo_data1/user_action.csv')
ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward"]
col = ["userid", "feedid", "date_", "device"] + ACTION_LIST

END_DAY = 15
start_day = 2

for start in range(start_day, END_DAY+1):
    print(start)
    print('---')
    
    tmp_his = history_data[history_data['date_']<start]
    tmp_his.to_csv('offline_test_'+'1_'+str(start-1)+'_log.csv',index=False)
    
    offline_test = history_data[history_data['date_']==start]
    offline_test[col].to_csv('offline_test_label_'+str(start)+'.csv',index=False)