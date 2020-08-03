#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time     :  2020/7/31
@Author   :  yangjing gan
@File     :  test.py
@Contact  :  ganyangjing95@qq.com
@License  :  (C)Copyright 2019-2020

'''
import numpy as np
import pandas as pd
from constrained_optimization import integrityOptimizationFactory, binaryOptimizationFactory, realNumberOptimizationFactory, on_optimize

if __name__ == '__main__':
    data = pd.read_excel('./integrity_optimization.xlsx', sep=' ', encoding='utf-8')
    decision_var_shape = [3, 4]
    
    # test1
    y = data.iloc[:, 0:4].values
    z = data.iloc[:, 4].values
    a = np.array([80, 140, 30, 50])
    b = np.array([30, 70, 10, 10])
    
    decision_var_symbol = ['x']
    obj_formula = 'sum(x * "data.iloc[:, 0:4].values")'  # min
    d_cons_formula = {'sum(x,0)-"np.array([30, 70, 10, 10]")': 'ge', '"np.array([80, 140, 30, 50])"-sum(x,0)': 'gt', '"data.iloc[:, 4].values"-sum(x,1)': 'eq', 'x': 'gt'}  # >=0  # >=0
    
    # decision_var_symbol = ['x']
    # obj_formula = 'sum(x * "y")'  # min
    # d_cons_formula = {'sum(x1,0)-"b"': 'ge', '"a"-sum(x1,0)': 'gt', '"z"-sum(x1,1)': 'eq', 'x1': 'gt'}  # >=0  # >=0
    
    # decision_var_symbol = ['x1', 'x2']
    # obj_formula = 'sum(x1 * "y" + x2 * "y")'  # min
    # d_cons_formula = {'sum(x1,0)-"b"': 'ge', '"a"-sum(x1,0)': 'gt', '"z"-sum(x1,1)': 'eq', 'x1': 'gt'}  # >=0  # >=0
    
    oop = on_optimize(
        integrityOptimizationFactory(decision_var_shape, decision_var_symbol, obj_formula, d_cons_formula, data))
    oop.show_info()
    oop = on_optimize(
        binaryOptimizationFactory(decision_var_shape, decision_var_symbol, obj_formula, d_cons_formula, data))
    oop.show_info()
    oop = on_optimize(
        realNumberOptimizationFactory(decision_var_shape, decision_var_symbol, obj_formula, d_cons_formula, data))
    oop.show_info()