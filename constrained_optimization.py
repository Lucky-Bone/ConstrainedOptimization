#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time     :  2020/7/22
@Author   :  yangjing gan
@File     :  constrained_optimization.py
@Contact  :  ganyangjing95@qq.com
@License  :  (C)Copyright 2019-2020

'''
import re
import torch
import numpy as np
import pandas as pd
from pulp import LpVariable, LpInteger, LpProblem, LpMinimize, lpSum
import six
from abc import abstractmethod, ABCMeta


def convert_func_op_format(formula, op_convert_map):
    """
    将公式对象中的运算符转为指定的类型的运算符
    :param formula: str，输入的公式，如'min("desc_var"）-0.01'
    :param convert_map: dict，运算符替换表，如{"sum(": "lpSum(",}
    :return: pulp可解析的公式对象
    """
    for original_op, converted_op in op_convert_map.items():
        formula = formula.replace(original_op, converted_op)
    return formula


@six.add_metaclass(ABCMeta)
# Abstract Factory
class optimizationFactory(metaclass = ABCMeta):
    @abstractmethod
    def solve(self):
        raise ValueError('subclass must implement this method!')
  
  
# 抽象产品
class pulpSolver(metaclass = ABCMeta):
    def __init__(self, decision_vars, obj_formula, l_cons_formula):
        """
        将字符串对象转为pulp对象
        :param formula: str，输入的公式，如'min("desc_var"）-0.01'
        :param convert_map: dict，运算符替换表，如{"sum(": "lpSum(",}
        :return: pulp可解析的公式对象
        """
        self.pulp_convert_map = {
            "sum(": "np.sum(",
            "avg(": "np.mean(",
            "exp(": "np.exp(",
            "log(": "np.log(",
        }
        self.decision_var_row = decision_vars[0]
        self.decision_var_col = decision_vars[1]
        self.obj_func = self.convert_pulp_func(obj_formula, self.pulp_convert_map)
        self.l_cons_func = [self.convert_pulp_func(cons_formula, self.pulp_convert_map) for cons_formula in l_cons_formula]
    
    def convert_pulp_func(self, formula, op_convert_map):
        """
        将字符串公式转为pulp格式的公式
        :param formula:
        :param op_convert_map:
        :return:
        """
        return convert_func_op_format(formula, op_convert_map)
        
    @abstractmethod
    def make_loss(self):
        """
        定义loss
        """
        raise ValueError('subclass must implement this method!')

    @abstractmethod
    def solve(self):
        """
        求解
        :return:
        """
        raise ValueError('subclass must implement this method!')

class GDSolver(metaclass = ABCMeta):
    def __init__(self, decision_vars, obj_formula, l_cons_formula):
        """
        将字符串对象转为torch对象
        :param formula: str，输入的公式，如'min("desc_var"）-0.01'
        :param convert_map: dict，运算符替换表，如{"sum(": "lpSum(",}
        :return: pulp可解析的公式对象
        """
    
        self.torch_convert_map = {
            "sum(": "torch.sum(",
            "avg(": "torch.mean(",
            "exp(": "torch.exp(",
            "log(": "torch.log(",
        }
        self.decision_var_row = decision_vars[0]
        self.decision_var_col = decision_vars[1]
        self.obj_func = self.convert_torch_func(obj_formula, self.torch_convert_map)
        self.l_cons_func = [self.convert_torch_func(cons_formula, self.torch_convert_map) for cons_formula in l_cons_formula]
    
    def convert_torch_func(self, formula, op_convert_map):
        """
        将字符串公式转为pulp格式的公式
        :param formula:
        :param op_convert_map:
        :return:
        """
        # 将公式对象中的运算符转为指定的类型的运算符
        formula = convert_func_op_format(formula, op_convert_map)
        # 将约束项中的非决策变量转为torch可识别对象
        pattern = '\"(' + re.escape('\s.*') + ')\"' + '([>=!<]*0?)'
        tensor_expr = lambda x: '(torch.from_numpy(' + x[1] + '"])' + x[2] + ').type(torch.FloatTensor)'
        formula = re.sub(pattern, tensor_expr, formula)
        return formula
        
    @abstractmethod
    def make_loss(self):
        """
        定义loss
        """
        raise ValueError('subclass must implement this method!')

    @abstractmethod
    def solve(self):
        """
        求解
        :return:
        """
        raise ValueError('subclass must implement this method!')

        
# 具体产品
class pulpIntegritySolver(pulpSolver):
    def __init__(self, decision_vars, obj_formula, l_cons_formula):
        super(pulpIntegritySolver, self).__init__(decision_vars, obj_formula, l_cons_formula)
        # generate decision_vars
        self.decision_vars = [LpVariable('dcs_var_{}'.format(i), cat = LpInteger) for i in range(self.decision_var_row*self.decision_var_col)]
        self.decision_vars = np.array(self.decision_vars).reshape([self.decision_var_row, self.decision_var_col])
        # define optimization object
        self.solver = LpProblem('Intergrity_Optimization', LpMinimize)
        
    def make_loss(self, ):
        """
        定义loss
        """
        x = self.decision_vars
        # object function
        self.solver += eval(self.obj_func)

        # constrain functions
        for func in self.l_cons_func:
            tmp_func = eval(func)
            if isinstance(tmp_func, np.ndarray):
                for ele in tmp_func:
                    self.solver += ele
            else:
                self.solver += tmp_func
    
    def solve(self):
        """
        求解
        :return:
        """
        status = self.solver.solve()
        return status
        
        
class GDRealNumberSolver(GDSolver):
    def __init__(self, decision_vars, obj_formula, l_cons_formula):
        super(GDRealNumberSolver, self).__init__(decision_vars, obj_formula, l_cons_formula)
        self.decision_vars = torch.from_numpy(np.random.random([self.decision_var_row, self.decision_var_col]))
     
    def convert_cons_to_uncons(self, cons_formula, cons_op):
        """
        将约束问题（包含等式约束和不等式约束）转为无约束问题
        :param cons_formula: 约束公式
        :param cons_op: 约束公式跟0的关系
        :return:
        """
        # 将约束转为惩罚项
        if cons_op in ["ge", "gt"]:
            cons_formula = '(torch.min(torch.tensor(0.0), {}))**2'.format(cons_formula)
        elif cons_op in ["eq"]:
            cons_formula = '({})**2'.format(cons_formula)
        else:
            pass
        return cons_formula
        
    def make_loss(self):
        """
        定义loss
        """
        x = self.decision_vars
        # object function
        self.loss = eval(self.obj_func)

        # constrain functions
        for func in self.l_cons_func:
            tmp_func = eval(func)
            if isinstance(tmp_func, np.ndarray):
                for ele in tmp_func:
                    self.loss += eval(self.convert_cons_to_uncons(ele, 'ge'))
            else:
                self.loss += tmp_func


    def solve(self):
        """
        求解
        :return:
        """

# 具体工厂
class integrityOptimizationFactory(optimizationFactory):
    def __init__(self, decision_vars: list, obj_formula: str, l_cons_formula: list, ):
        self.pulp_intergrity_solver = pulpIntegritySolver(decision_vars, obj_formula, l_cons_formula)
        self.loss = self.pulp_intergrity_solver.make_loss()
    
    def solve(self):
        return self.pulp_intergrity_solver.solve()


class realNumberOptimizationFactory(optimizationFactory):
    def __init__(self, decision_vars: list, obj_formula: str, l_cons_formula: list, ):
        self.GD_realNum_solver = GDRealNumberSolver(decision_vars, obj_formula, l_cons_formula)
        self.loss = self.GD_realNum_solver.make_loss()
    
    def solve(self):
        return self.GD_realNum_solver.solve()

# 客户端
class constrainedOptimization(object):
    def __init__(self, status):
        """
        初始化
        """
        self.status = status

    def show_info(self):
        """
        打印求解结果
        :return:
        """
        print("Solve Result:")
        print(self.status)

        
def on_optimize(optimization_factory):
    status = optimization_factory.solve()
    return constrainedOptimization(status)


data = pd.read_excel('./integrity_optimization.xlsx', sep = ' ', encoding = 'utf-8')
decision_var_shape = [3, 4]

y = data.iloc[:, 0:4].values
z = data.iloc[:, 4].values
a = [80, 140, 30, 50]
b = [30, 70, 10, 10]
obj_formula = 'sum(x * y)' #min
l_cons_formula = ['sum(x,0)-b','a-sum(x,0)', 'z-sum(x,1)']  # >=0
# oop = on_optimize(integrityOptimizationFactory(decision_var_shape, obj_formula, l_cons_formula))
# oop.show_infonfo()

obj_formula = 'sum(x * "y")' #min
oop = on_optimize(realNumberOptimizationFactory(decision_var_shape, obj_formula, l_cons_formula))
