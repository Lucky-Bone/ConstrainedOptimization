#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time     :  2020/9/7
@Author   :  yangjing gan
@File     :  lgb_graphviz.py
@Contact  :  ganyangjing95@qq.com
@License  :  (C)Copyright 2019-2020

'''

import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('数据...')
x_train = np.random.random((1000,10))
y_train = np.random.rand(1000)>0.5
x_test = np.random.random((100,10))
y_test = np.random.randn(100)>0.5

# 导入到lightgbm矩阵
lgb_train = lgb.Dataset(x_train, y_train)
lgb_test = lgb.Dataset(x_test, y_test, reference=lgb_train)

# 设置参数
params = {
    'num_leaves': 5,
    'metric': ('auc', 'logloss'),#可以设置多个评价指标
    'verbose': 0
}
# if (evals_result and gbm) not in locbals():
	# global evals_result,gbm
#如果是局部变量的话，推荐把他们变成全局变量，这样plot的代码位置不受限制
evals_result = {}  #记录训练结果所用

print('开始训练...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=[lgb_train, lgb_test],
                evals_result=evals_result,#非常重要的参数,一定要明确设置
                verbose_eval=10)

# print('画出训练结果...')
# ax = lgb.plot_metric(evals_result, metric='auc')#metric的值与之前的params里面的值对应
# plt.show()
#
# print('画特征重要性排序...')
# ax = lgb.plot_importance(gbm, max_num_features=10)#max_features表示最多展示出前10个重要性特征，可以自行设置
# plt.show()
#
# print('Plot 3th tree...')  # 画出决策树，其中的第三颗
# ax = lgb.plot_tree(gbm, tree_index=3, figsize=(20, 8), show_info=['split_gain'])
# plt.show()

print('导出决策树的pdf图像到本地')#这里需要安装graphviz应用程序和python安装包
graph = la = lgb.create_tree_digraph(gbm, tree_index=3, name='Tree3')
graph.render(view=True)
from graphviz_sql.extract import to_sql
to_sql('Tree3.gv', 'Tree3.sql')