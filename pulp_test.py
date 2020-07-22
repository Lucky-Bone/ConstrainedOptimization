#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time     :  2020/7/2
@Author   :  yangjing gan
@File     :  pulp_test.py
@Contact  :  ganyangjing95@qq.com
@License  :  (C)Copyright 2019-2020

'''

from pulp import *


def get_re():
    pass


def getresult(c, con):

# 设置对象
    prob = LpProblem('myPro', LpMinimize)
# 设置三个变量，并设置变量最小取值

    x11 = LpVariable("x11", lowBound=0)
    x12 = LpVariable("x12", lowBound=0)
    x13 = LpVariable("x13", lowBound=0)
    x14 = LpVariable("x14", lowBound=0)
    x21 = LpVariable("x21", lowBound=0)
    x22 = LpVariable("x22", lowBound=0)
    x23 = LpVariable("x23", lowBound=0)
    x24 = LpVariable("x24", lowBound=0)
    x31 = LpVariable("x31", lowBound=0)
    x32 = LpVariable("x32", lowBound=0)
    x33 = LpVariable("x33", lowBound=0)
    x34 = LpVariable("x34", lowBound=0)

    X = [x11, x12, x13, x14, x21, x22, x23, x24, x31, x32, x33, x34]

    #c = [160, 130, 220, 170, 140, 130, 190, 150, 190, 200, 230]






# 目标函数
    z = 0
    for i in range(len(X)):
        z += X[i]*c[i]
    #print(z)
    prob += z

# 载入约束变量
    prob += x11+x12+x13+x14 == con[0]# 约束条件1
    prob += x21+x22+x23+x24 == con[1]
    prob += x31+x32+x33+x34 == con[2]

    prob += x11+x21+x31 <= con[3]
    prob += x11+x21+x31 >= con[4]

    prob += x12 + x22 + x32 <= con[5]
    prob += x12 + x22 + x32 >= con[6]

    prob += x13 + x23 + x33 <= con[7]
    prob += x13 + x23 + x33 >= con[8]
    prob += x14 + x24 + x34 <= con[9]
    prob += x14 + x24 + x34 >= con[10]

# 求解

    status = prob.solve()

    print(status)
    print(LpStatus[status])
    print(value(prob.objective))  # 计算结果


# 显示结果
#     for i in prob.variables():
#         print(i.name + "=" + str(i.varValue))
    for i in prob.variables():
        print(i.varValue)


if __name__ == '__main__':
    c = [160, 130, 220, 170, 140, 130, 190, 150, 190, 200, 230, 20000]
    con = [50, 60, 50, 80, 30, 140, 70, 30,10, 50, 10]
    getresult(c, con)