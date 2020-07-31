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
from tqdm import trange
from pulp import LpVariable, LpInteger, LpProblem, LpMinimize, lpSum, pulp, LpStatus
from torch.optim.lr_scheduler import ReduceLROnPlateau
from abc import abstractmethod, ABCMeta
import six
# @six.add_metaclass(ABCMeta)


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


# Abstract Factory
class optimizationFactory(metaclass = ABCMeta):
    @abstractmethod
    def solve(self):
        raise ValueError('subclass must implement this method!')
  
  
# 抽象产品
class pulpSolver(metaclass = ABCMeta):
    @abstractmethod
    def __init__(self, decision_var_shape, decision_var_symbol, obj_formula, d_cons_formula):
        """
        将字符串对象转为pulp对象
        """
        self.pulp_convert_map = {
            "sum(": "np.sum(",
            "avg(": "np.mean(",
            "exp(": "np.exp(",
            "log(": "np.log(",
        }
        self.pulp_solve_status = {
            0: 'Not Solved', # Not Solved
            1: 'Optimal', # Optimal
            -1: 'Infeasible', # Infeasible
            -2: 'Unbounded', # Unbounded
            -3: 'Undefined'} # Undefined
        self.decision_var_row = decision_var_shape[0]
        self.decision_var_col = decision_var_shape[1]
        self.decision_var_symbol = decision_var_symbol
        self.obj_func = self.convert_pulp_func(obj_formula, self.pulp_convert_map)
        self.d_cons_func = {self.convert_pulp_func(cons_formula, self.pulp_convert_map): cons_op for cons_formula, cons_op
                        in d_cons_formula.items()}

    def convert_pulp_func(self, formula, op_convert_map):
        """
        将字符串公式转为pulp格式的公式
        :param formula: str，输入的公式，如'min("desc_var"）-0.01'
        :param convert_map: dict，运算符替换表，如{"sum(": "lpSum(",}
        :return: pulp可解析的公式对象
        """
        # 将公式对象中的运算符转为指定的类型的运算符
        formula = convert_func_op_format(formula, op_convert_map)
        # 将公式中的非决策变量转为pulp可识别对象
        pattern = '\"(.*?)\"([>=!<]*0?)'
        tensor_expr = lambda x: x[1] + x[2]
        formula = re.sub(pattern, tensor_expr, formula)
        return formula

    def convert_cons(self, cons_formula, cons_op):
        """
        将约束项中pulp格式的公式和运算符结合起来
        :param cons_formula: 约束公式
        :param cons_op: 约束公式跟0的关系
        :return:
        """
        # 将约束转为惩罚项
        if cons_op in ["ge", "gt"]:
            cons_formula = cons_formula >= 0
        elif cons_op in ["eq"]:
            cons_formula = cons_formula == 0
        else:
            pass
        return cons_formula

    def make_loss(self, ):
        """
        定义求解器solver
        """
        for attr_name, attr_value in self.d_decision_vars.items():
            exec("{} = attr_value".format(attr_name))
        # object function
        self.solver += eval(self.obj_func)
    
        # constrain functions
        for cons_formula, cons_op in self.d_cons_func.items():
            cons_func = eval(cons_formula)
            if isinstance(cons_func, np.ndarray) and len(cons_func.shape) == 2:
                for cons_item in cons_func:
                    for ele in cons_item:
                        self.solver += self.convert_cons(ele, cons_op)
            elif isinstance(cons_func, np.ndarray) and len(cons_func.shape) == 1:
                for ele in cons_func:
                    self.solver += self.convert_cons(ele, cons_op)
            else:
                self.solver += self.convert_cons(cons_func, cons_op)

    def solve(self):
        """
        求解
        :return:
        """
        status = self.pulp_solve_status[self.solver.solve()]
        objective = pulp.value(self.solver.objective)
        decision_var = []
        for v in self.solver.variables():
            # print(v.name, v.varValue)
            decision_var.append(v.varValue)
        decision_var = np.array(decision_var).reshape([len(self.decision_var_symbol), self.decision_var_row, self.decision_var_col])
        return status, objective, decision_var

class GDSolver(metaclass = ABCMeta):
    def __init__(self, decision_var_shape, decision_var_symbol, obj_formula, d_cons_formula):
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
        self.decision_var_row = decision_var_shape[0]
        self.decision_var_col = decision_var_shape[1]
        self.decision_var_symbol = decision_var_symbol
        self.obj_formula = self.convert_torch_func(obj_formula, self.torch_convert_map)
        self.d_cons_formula = {self.convert_torch_func(cons_formula, self.torch_convert_map): cons_op for cons_formula, cons_op in d_cons_formula.items()}
    
    def convert_torch_func(self, formula, op_convert_map):
        """
        将字符串公式转为torch格式的公式
        :param formula:
        :param op_convert_map:
        :return:
        """
        # 将公式对象中的运算符转为指定的类型的运算符
        formula = convert_func_op_format(formula, op_convert_map)
        # 将公式中的非决策变量转为torch可识别对象
        pattern = '\"(.*?)\"([>=!<]*0?)'
        tensor_expr = lambda x: '(torch.from_numpy(' + x[1] + ')' + x[2] + ').type(torch.FloatTensor)'
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
    def __init__(self, decision_var_shape, decision_var_symbol, obj_formula, d_cons_formula):
        super(pulpIntegritySolver, self).__init__(decision_var_shape, decision_var_symbol, obj_formula, d_cons_formula)
        # generate decision_vars
        self.d_decision_vars = dict()
        for var in self.decision_var_symbol:
            self.d_decision_vars[var] = [LpVariable('{}_{}_{}'.format(var, i, j), cat=LpInteger) for i in range(self.decision_var_row) for
                  j in range(self.decision_var_col)]
            self.d_decision_vars[var] = np.array(self.d_decision_vars[var]).reshape([self.decision_var_row, self.decision_var_col])
        # define optimization object
        self.solver = LpProblem('Intergrity_Optimization', LpMinimize)
  
class pulpBinarySolver(pulpSolver):
    def __init__(self, decision_var_shape, decision_var_symbol, obj_formula, d_cons_formula):
        super(pulpBinarySolver, self).__init__(decision_var_shape, decision_var_symbol, obj_formula, d_cons_formula)
        # generate decision_vars
        self.d_decision_vars = dict()
        for var in self.decision_var_symbol:
            self.d_decision_vars[var] = [LpVariable('{}_{}_{}'.format(var, i, j), cat=LpInteger) for i in
                                         range(self.decision_var_row) for
                                         j in range(self.decision_var_col)]
            self.d_decision_vars[var] = np.array(self.d_decision_vars[var]).reshape(
                [self.decision_var_row, self.decision_var_col])
        # define optimization object
        self.solver = LpProblem('Intergrity_Optimization', LpMinimize)
        
class GDRealNumberSolver(GDSolver):
    def __init__(self, decision_var_shape: list, decision_var_symbol: list, obj_formula: str, d_cons_formula: dict):
        super(GDRealNumberSolver, self).__init__(decision_var_shape, decision_var_symbol, obj_formula, d_cons_formula)
        # generate decision_vars
        self.decision_vars = torch.from_numpy(np.random.random([len(self.decision_var_symbol), self.decision_var_row, self.decision_var_col])).type(torch.FloatTensor)
        self.lr = 0.001
        self.n_epoch = 1000
        self.stage_num = 100
        self.patience = 8  # for early stopping
        self.lr_schedule = True
        self.delta = 1e-4  # threshold of loss difference
    
    def convert_cons_to_uncons(self, cons_formula, cons_op):
        """
        将约束项中torch格式的公式（包含等式约束和不等式约束）转为无约束问题
        :param cons_formula: 约束公式
        :param cons_op: 约束公式跟0的关系
        :return:
        """
        # 将约束转为惩罚项
        if cons_op in ["ge", "gt"]:
            cons_formula = 'torch.sum((torch.min(torch.tensor(0.0), {}))**2)'.format(cons_formula)
        elif cons_op in ["eq"]:
            cons_formula = 'torch.sum(({})**2)'.format(cons_formula)
        else:
            pass
        return cons_formula
        
    def make_loss(self):
        """
        定义loss
        """
        self.obj_func = eval('lambda {}: {}'.format(','.join(self.decision_var_symbol), self.obj_formula))
        # constrain functions
        l_cons_formula = list()
        for cons_formula, cons_op in self.d_cons_formula.items():
            cons_formula = self.convert_cons_to_uncons(cons_formula, cons_op)
            try:
                cons_val = eval('lambda {}: {}'.format(','.join(self.decision_var_symbol), cons_formula))
            except SyntaxError:
                raise SyntaxError('Invalid constrain format: {}'.format(cons_formula))
            if not isinstance(cons_val(*self.decision_vars), torch.Tensor):
                raise SyntaxError('Invalid constrain value: {}, the values returns supposed to be a single value not array'.format(cons_formula))
            l_cons_formula.append(cons_formula)
        self.cons_formula = '+'.join(l_cons_formula)
        self.cons_func = eval('lambda {}: {}'.format(','.join(self.decision_var_symbol), self.cons_formula))
        self.loss_formula = 'lambda {}, r: {} + r * ({})'.format(','.join(self.decision_var_symbol), self.obj_formula, '+'.join(l_cons_formula))
        self.loss_func = eval(self.loss_formula)
        
    def solve(self):
        """
        求解
        :return:
        """
        # calculate initial value
        self.decision_vars = torch.autograd.Variable(torch.rand([len(self.decision_var_symbol), self.decision_var_row, self.decision_var_col]), requires_grad=True)
        ini_obj_value = self.obj_func(*self.decision_vars)
        ini_cons_value = self.cons_func(*self.decision_vars)
        self.r = torch.autograd.Variable(ini_obj_value / (ini_cons_value + 1e-5), requires_grad=False)
        
        # solve
        status = 1 # 求解状态
        stage_early_stop_count = 0
        optimizer = torch.optim.Adam([self.decision_vars, self.r], lr = self.lr, amsgrad=False, eps=1e-6)
        lr_schedule = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience = self.patience, min_lr = 1e-7, verbose=True)
        init_record = np.inf
        for stage in range(self.stage_num):
            record = [init_record]
            early_stop_count = 0
            with trange(self.n_epoch) as epoch_bar:
                epoch_bar.set_description("Stage %s training" % (stage))
                for epoch in epoch_bar:
                    optimizer.zero_grad()
                    loss = self.loss_func(*self.decision_vars, self.r)
                    loss.backward()
                    optimizer.step()
                    record.append(loss.cpu().detach().numpy())
                    
                    if (epoch + 1) % 100 == 0:
                        print(epoch+1, self.r, loss)
                        
                    if record[-1] >= min(record[:-1]):
                        early_stop_count += 1
                        if early_stop_count >= self.patience:
                            stage_early_stop_count += 1
                            epoch_bar.update(epoch)
                            epoch_bar.close()
                            break
                    else:
                        early_stop_count = 0
                epoch_bar.close()
                
            if stage_early_stop_count > self.patience and self.cons_func(*self.decision_vars) <= 0.05:
                print("Early Stopping")
                break
            else:
                self.r *= 2
                lr_schedule.step(loss)
        
        if stage == self.stage_num-1:
            status = 0
            print("Failed to converge!")
        
        objective = self.obj_func(*self.decision_vars)
        decision_vars = self.decision_vars.cpu().detach().numpy()
        return status, objective, decision_vars
    

# 具体工厂
class integrityOptimizationFactory(optimizationFactory):
    def __init__(self, decision_var_shape: list, decision_var_symbol: list, obj_formula: str, d_cons_formula: dict,):
        self.pulp_intergrity_solver = pulpIntegritySolver(decision_var_shape, decision_var_symbol, obj_formula, d_cons_formula)
        self.loss = self.pulp_intergrity_solver.make_loss()
        print(self.pulp_intergrity_solver.solver)
    
    def solve(self):
        return self.pulp_intergrity_solver.solve()

class binaryOptimizationFactory(optimizationFactory):
    def __init__(self, decision_var_shape: list, decision_var_symbol: list, obj_formula: str, d_cons_formula: dict,):
        self.pulp_binary_solver = pulpBinarySolver(decision_var_shape, decision_var_symbol, obj_formula, d_cons_formula)
        self.loss = self.pulp_binary_solver.make_loss()
    
    def solve(self):
        return self.pulp_binary_solver.solve()
    
class realNumberOptimizationFactory(optimizationFactory):
    def __init__(self, decision_var_shape: list, decision_var_symbol: list, obj_formula: str, d_cons_formula: dict, ):
        self.GD_realNum_solver = GDRealNumberSolver(decision_var_shape, decision_var_symbol, obj_formula, d_cons_formula)
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
a = np.array([80, 140, 30, 50])
b = np.array([30, 70, 10, 10])

decision_var_symbol = ['x']
obj_formula = 'sum(x * "y")' #min
d_cons_formula = {'sum(x,0)-"b"': 'ge','"a"-sum(x,0)': 'gt', '"z"-sum(x,1)': 'eq', 'x':'gt'}  # >=0  # >=0

decision_var_symbol = ['x1', 'x2']
obj_formula = 'sum(x1 * "y" + x2 * "y")' #min
d_cons_formula = {'sum(x1,0)-"b"': 'ge','"a"-sum(x1,0)': 'gt', '"z"-sum(x1,1)': 'eq', 'x1':'gt'}  # >=0  # >=0

oop = on_optimize(integrityOptimizationFactory(decision_var_shape, decision_var_symbol, obj_formula, d_cons_formula))
oop.show_info()
oop = on_optimize(binaryOptimizationFactory(decision_var_shape, decision_var_symbol, obj_formula, d_cons_formula))
oop.show_info()
oop = on_optimize(realNumberOptimizationFactory(decision_var_shape, decision_var_symbol, obj_formula, d_cons_formula))
oop.show_info()
