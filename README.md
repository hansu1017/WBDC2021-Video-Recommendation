# WBDC2021-Video-Recommendation
2021微信大数据挑战赛树模型初赛方案

赛题：https://algo.weixin.qq.com/2021/problem-description

方案和经验总结blog：https://zhuanlan.zhihu.com/p/402162597

# 代码流程
generate_sample.py：生成样本

generate_features.py：生成树模型特征

prepare_for_tree.py：整合树模型特征

offline_for_lgb.py：训练lightgbm模型

offline_for_cbt.py：训练catboost模型
