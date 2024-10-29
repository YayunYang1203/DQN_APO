# -*- coding: utf-8 -*-
"""
CEC 测试函数 系列
@author: 微信公众号：优化算法侠，Swarm-Opti

"""

try:
    # Check if running inside IPython
    from IPython import get_ipython

    if get_ipython() is not None:
        get_ipython().magic('reset -sf')  # Clear all variables
except ImportError:
    pass  # Handle the case where IPython is not installed or not available

import numpy as np
from matplotlib import pyplot as plt

# In[]:
import opfunu  # 参考文档：https://github.com/thieu1995/opfunu
import mealpy
from mealpy.swarm_based import WOA, GWO
from mealpy import get_optimizer_by_name
from APO_func import APO_func



'''
适应度函数及维度dim的选择
cec2022：F1-F12, 可选 dim = 2, 10, 20

'''
fun_name = 'F1'  # 按需修改
year = '2022'  # 按需修改
func_num = fun_name + year
dim = 20  # 维度，根据cec函数选择对应维度

'''定义cec函数'''
def cec_fun(x):
    funcs = opfunu.get_functions_by_classname(func_num)
    func = funcs[0](ndim=dim)
    F = func.evaluate(x)
    return F

''' fit_func->目标函数, lb->下限, ub->上限 '''
problem_dict = {
    "fit_func": cec_fun,
    "lb": opfunu.get_functions_by_classname(func_num)[0](ndim=dim).lb.tolist(),
    "ub": opfunu.get_functions_by_classname(func_num)[0](ndim=dim).ub.tolist(),
    "minmax": "min",
}

''' 调用优化算法 '''
epoch = 100  # 最大迭代次数
pop_size = 50  # 种群数量

''' 第一种方式，需：from mealpy.swarm_based import WOA,GWO '''
# woa_model = WOA.OriginalWOA(epoch, pop_size)
# gwo_model = GWO.OriginalGWO(epoch, pop_size)
''' 第二种方式，需：from mealpy import get_optimizer_by_name'''
woa_model = get_optimizer_by_name("OriginalWOA")(epoch, pop_size)
gwo_model = get_optimizer_by_name("OriginalGWO")(epoch, pop_size)
bestFit_APO, record_time_APO, bestProtozoa_APO, best_learning_rate_APO, best_epsilon_APO, fitness_history = APO_func(1, 2, 100, 200, -100, 100)

def cec_fun(x):
    funcs = opfunu.get_functions_by_classname(func_num)
    func = funcs[0](ndim=dim)
    F = func.evaluate(x)
    return x, F


'''求解cec函数'''
woa_best_x, woa_best_f = woa_model.solve(problem_dict)
print(f"WOA最优解: {woa_best_x}, \nWOA最优函数值: {woa_best_f}")
gwo_best_x, gwo_best_f = gwo_model.solve(problem_dict)
print(f"GWO最优解: {gwo_best_x}, \nGWO最优函数值: {gwo_best_f}")
cec_best_x_APO, cec_best_f_APO = cec_fun(bestProtozoa_APO)
print(f"APO最优解: {bestProtozoa_APO}, \nAPO最优函数值: {bestFit_APO}")




''' 
    绘制适应度曲线
    model.history.list_global_best_fit：适应度曲线
'''
plt.figure
# plt.semilogy(Curve,'r-',linewidth=2)

# 绘制适应度曲线


plt.figure()
plt.plot(woa_model.history.list_global_best_fit, 'r-', linewidth=2, label='WOA')
plt.plot(gwo_model.history.list_global_best_fit, 'b-', linewidth=2, label='GWO')
APO_results = APO_func('fhd', 'Fid', 'dim', 'pop_size', 'iter_max', 'Xmin', 'Xmax', 'runid')
# 使用 NumPy 数组的方法访问数据
APO_results_array = np.array(APO_results[0])

# 现在可以使用转换后的 NumPy 数组进行绘图
plt.plot(APO_results_array, 'c-', linewidth=2, label='APO')



plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.grid()
plt.title('Convergence curve: ' + 'cec' + year + '-' + fun_name + ', Dim=' + str(dim))
plt.legend()
plt.savefig('convergence_curve.png')  # 保存为png格式
plt.show()

# In[]:

''' 绘制三维函数图 '''
# 仅修改n_space 和 show -> 可视化选择参数
plt.figure()
opfunu.plot_3d(opfunu.get_functions_by_classname(func_num)[0](ndim=2), n_space=500, show=True)
plt.title('cec' + year + '-' + fun_name)
plt.savefig('3d_function_plot.png')  # 保存为png格式
plt.show()
