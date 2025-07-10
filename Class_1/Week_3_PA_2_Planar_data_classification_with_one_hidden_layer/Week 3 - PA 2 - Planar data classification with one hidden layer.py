# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 13:42:07 2018

@author: Oscar

带有一个隐藏层的平面数据分类
"""

"""
欢迎来到您的第3周的编程任务。 
现在建立你的第一个神经网络，它将有一个隐藏层。 你会发现这个模型和你使用逻辑回归实现的模型有很大的区别。

你将学到如何：

实现具有单个隐藏层的2类分类神经网络
使用具有非线性激活功能的单位，例如tanh
计算交叉熵损失
实现向前和向后传播
"""

"""
首先导入您在此作业期间需要的所有软件包。

numpy是用Python进行科学计算的基本软件包。
sklearn为数据挖掘和数据分析提供了简单高效的工具。
matplotlib是一个用于在Python中绘制图表的库。
testCases提供了一些测试示例来评估函数的正确性
planar_utils提供了在这个任务中使用的各种有用的功能
"""

import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1) # set a seed so that the results are consistent

"""
首先，让我们来看看你将要使用的数据集。 下面的代码会将一个“花”2类数据集加载到变量X和Y中。
"""

X , Y = load_planar_dataset()
"""
使用matplotlib可视化数据集。 
数据看起来像一朵红色（标签y = 0）和一些蓝色（y = 1）点的“花朵”。 你的目标是建立一个模型来适应这些数据。
"""
#绘制散点图
#plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral) #figure_1

"""
- 包含特征（x1，x2）的numpy数组（矩阵）X
- 包含标签（红色：0，蓝色：1）的numpy数组（矢量）Y.
"""

shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]
print ("X的维度为: " + str(shape_X))
print ("Y的维度为: " + str(shape_Y))
print ("数据集里面的数据有：" + str(m) + " 个")


"""
- 简单的Logistic回归
在构建完整的神经网络之前，先让我们看看逻辑回归在这个问题上的表现。
 你可以使用sklearn的内置函数来做到这一点。 运行下面的代码来训练数据集上的逻辑回归分类器。
"""

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T,Y.T)

#plot_decision_boundary(lambda x: clf.predict(x), X, Y)
#plt.title("Logistic Regression")

LR_predictions  = clf.predict(X.T)
print ('逻辑回归的准确性： %d ' % float((np.dot(Y, LR_predictions) + 
		np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) +
       '% ' + "(正确标记的数据点所占的百分比)")
"""
逻辑回归的准确性：47％（正确标记的数据点的百分比）
解释：数据集不是线性可分的，所以逻辑回归表现不佳。 希望神经网络能做得更好。 现在我们来试试吧！

"""


"""
Logistic回归在“花卉数据集”上效果不佳。 你将用一个隐藏层训练一个神经网络。
#classification_kiank.png

Mathematically:

For one example  x(i)x(i) :
z[1](i)=W[1]x(i)+b[1](i)(1)
(1)z[1](i)=W[1]x(i)+b[1](i)
 
a[1](i)=tanh(z[1](i))(2)
(2)a[1](i)=tanh⁡(z[1](i))
 
z[2](i)=W[2]a[1](i)+b[2](i)(3)
(3)z[2](i)=W[2]a[1](i)+b[2](i)
 
ŷ (i)=a[2](i)=σ(z[2](i))(4)
(4)y^(i)=a[2](i)=σ(z[2](i))
 
y(i)prediction={10if a[2](i)>0.5otherwise (5)
(5)yprediction(i)={1if a[2](i)>0.50otherwise 
 
Given the predictions on all the examples, you can also compute the cost  JJ  as follows:
J=−1m∑i=0m(y(i)log(a[2](i))+(1−y(i))log(1−a[2](i)))(6)
(6)J=−1m∑i=0m(y(i)log⁡(a[2](i))+(1−y(i))log⁡(1−a[2](i)))

"""

"""
构建神经网络的一般方法是：

1.定义神经网络结构（输入单元的数量，隐藏单元的数量等）。
2.初始化模型的参数
3.循环：
     - 实施前向传播
     - 计算损失
     - 实现向后传播以获得渐变
     - 更新参数（梯度下降）
     
您经常构建帮助函数来计算步骤1-3，
然后将它们合并到一个函数中，我们称之为nn_model（）。 
一旦你建立了nn_model（）并学习了正确的参数，你就可以预测新的数据。

4.1 - 定义神经网络结构
练习：定义三个变量：

- n_x：输入图层的大小
- n_h：隐藏层的大小（将其设置为4）
- n_y：输出图层的大小
提示：使用X和Y的形状来查找n_x和n_y。 另外，将隐藏层大小硬编码为4。
     
"""


def layer_size(X , Y):
    """
    参数：
     X - 输入数据集,维度为（输入大小，训练/测试的数量）
     Y - 标签，维度为（输出大小，训练/测试数量）
    
    返回：
     n_x - 输入图层的大小
     n_h - 隐藏层的大小
     n_y - 输出图层的大小
    """
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    
    return (n_x,n_h,n_y)

#测试layer_size
print("=========================测试layer_size=========================")
X_asses , Y_asses = layer_sizes_test_case()
(n_x,n_h,n_y) =  layer_size(X_asses,Y_asses)
print("输入层的大小为: n_x = " + str(n_x))
print("隐藏层的大小为: n_h = " + str(n_h))
print("输出层的大小为: n_y = " + str(n_y))


"""
4.2 - 初始化模型的参数
练习：实现函数initialize_parameters（）。

说明：

确保你的参数大小合适。 如果需要，请参考上面的神经网络图。
您将用随机值初始化权重矩阵。
使用：np.random.randn（a，b）* 0.01来随机初始化一个维度为（a，b）的矩阵。
你将初始化偏向量为零。
使用：np.zeros（（a，b））用零初始化形状矩阵（a，b）。
"""

def initialize_parameters( n_x , n_h ,n_y):
    """
    参数：
        n_x - 输入图层的大小
        n_h - 隐藏层的大小
        n_y - 输出层的大小
    
    返回：
        params - 包含你的参数的python字典：
            W1 - 权重矩阵,维度为（n_h，n_x）
            b1 - 偏向量，维度为（n_h，1）
            W2 - 权重矩阵，维度为（n_y，n_h）
            b2 - 偏向量，维度为（n_y，1）

    """
    np.random.seed(2) #建立一个种子，以便你的输出与我们的匹配，尽管初始化是随机的。
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))
    
    #使用断言确保我的数据格式是正确的
    assert(W1.shape == ( n_h , n_x ))
    assert(b1.shape == ( n_h , 1 ))
    assert(W2.shape == ( n_y , n_h ))
    assert(b2.shape == ( n_y , 1 ))
    
    parameters = {"W1" : W1,
              "b1" : b1,
              "W2" : W2,
              "b2" : b2 }
    
    return parameters

#测试initialize_parameters
print("=========================测试initialize_parameters=========================")    
n_x , n_h , n_y = initialize_parameters_test_case()
parameters = initialize_parameters(n_x , n_h , n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


"""
4.3 - 循环
问题：实现forward_propagation（）。

说明：

看看你的分类器的数学表示。
您可以使用函数sigmoid（）。 
你可以使用函数np.tanh（）。 
您必须执行的步骤是：

通过使用字典“参数”（它是initialize_parameters（）的输出）检索每个参数。
实现向前传播。 计算Z [1]，A [1]，Z [2] Z [1]，A [1]，Z [2]和A [2] A [2]（所有例子的预测向量 训练集）。
反向传播所需的值存储在“缓存”中。 缓存将作为反向传播函数的输入。
"""

def forward_propagation( X , parameters ):
    """
    参数：
         X - 维度为（n_x，m）的输入数据。
         parameters - 包含你的参数的python字典（初始化函数的输出）
    
    返回：
         A2 - 使用sigmoid()函数计算的第二次激活的数值
         缓存 - 包含“Z1”，“A1”，“Z2”和“A2”的字典
     """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    #前向传播计算A2
    Z1 = np.dot(W1 , X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2 , A1) + b2
    A2 = sigmoid(Z2)
     
    assert(A2.shape == (1,X.shape[1]))
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return (A2, cache)

#测试forward_propagation
print("=========================测试forward_propagation=========================") 
X_assess, parameters = forward_propagation_test_case()
A2, cache = forward_propagation(X_assess, parameters)
print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))


"""
J=−1m∑i=0m(y(i)log(a[2](i))+(1−y(i))log(1−a[2](i)))(13)
(13)J=−1m∑i=0m(y(i)log⁡(a[2](i))+(1−y(i))log⁡(1−a[2](i)))
"""

"""
练习：执行compute_cost（）来计算成本JJ的值。

说明：

有很多方法来实现交叉熵损失。 为了帮助你，我们给你如何实现-Σi= 0my（i）log（a [2]（i）） - Σi= 0my（i）log⁡（a [2]（i））：
logprobs = np.multiply（np.log（A2），Y）
成本= - np.sum（logprobs）＃不需要使用for循环！
（您可以使用np.multiply（），然后使用np.sum（）或直接使用np.dot（））。
"""

def compute_cost(A2,Y,parameters):
    """
    计算方程（13）中给出的交叉熵成本，
    
    参数：
         A2 - 使用sigmoid()函数计算的第二次激活的数值
         Y - “真实”标签矢量,维度为（1，示例数）
         parameters - 包含您的参数W1，B1，W2和B2的Python字典
    
    返回：
         成本 - 交叉熵成本给出方程（13）
    """
    
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    #计算成本
    logprobs = logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = - np.sum(logprobs) / m
    cost = float(np.squeeze(cost))
    
    assert(isinstance(cost,float))
    
    return cost
 
print("=========================测试compute_cost=========================") 
A2 , Y_assess , parameters = compute_cost_test_case()
print("cost = " + str(compute_cost(A2,Y_assess,parameters)))

"""
使用正向传播期间计算的高速缓存，现在可以实现反向传播。

问题：实现函数backward_propagation（）。

说明：反向传播通常是深度学习中最难（最具数学意义的）部分。 
为了帮助你，这里再次是反向传播讲座的幻灯片。 
由于您正在构建矢量化实现，因此您将需要使用此幻灯片右侧的六个方程。
"""

def backward_propagation(parameters,cache,X,Y):
    """
    使用上述说明实施反向传播。
    
    参数：
     parameters - 包含我们的参数的一个字典类型的变量。
     cache - 包含“Z1”，“A1”，“Z2”和“A2”的字典类型的变量。
     X - 输入数据，维度为（2，数量）
     Y - “True”标签，维度为（1，数量）
    
    返回：
     grads - 包含你的梯度相对于不同参数的python字典
    """
    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2= A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads
    
print("=========================测试backward_propagation=========================")
parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))

def update_parameters(parameters,grads,learning_rate=1.2):
    """
    使用上面给出的梯度下降更新规则更新参数
    
    参数：
     parameters - 包含你的参数的python字典
     grads - 包含你的渐变的python字典
     learning_rate - 学习速率
    
    返回：
     parameters - 包含更新参数的python字典
    """
    W1,W2 = parameters["W1"],parameters["W2"]
    b1,b2 = parameters["b1"],parameters["b2"]
    
    dW1,dW2 = grads["dW1"],grads["dW2"]
    db1,db2 = grads["db1"],grads["db2"]
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
#测试update_parameters
print("=========================测试update_parameters=========================")
parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

def nn_model(X,Y,n_h,num_iterations,print_cost=False):
    """
    参数：
        X - 形状数据集（2，示例数）
        Y - 形状标签（1，示例数）
        n_h - 隐藏层的大小
        num_iterations - 渐变下降循环中的迭代次数
        print_cost - 如果为True，则每1000次迭代打印一次成本
    
    返回：
        parameters - 模型学习的参数。 然后他们可以用来预测。
     """
     
    np.random.seed(3)
    n_x,n_y = layer_size(X,Y)[0],layer_size(X,Y)[2]
    parameters = initialize_parameters(n_x,n_h,n_y)
    W1,W2,b1,b2 = parameters["W1"],parameters["W2"],parameters["b1"],parameters["b2"]
    
    for i in range(num_iterations):
        A2 , cache = forward_propagation(X,parameters)
        cost = compute_cost(A2,Y,parameters)
        grads = backward_propagation(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads,learning_rate = 1.2)
        
        if print_cost:
            if i%1000 == 0:
                print("第 ",i," 次循环，成本为："+str(cost))
    return parameters
        
print("=========================测试nn_model=========================")
X_assess, Y_assess = nn_model_test_case()

parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=False)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


def predict(parameters,X):
    """
    使用学习的参数，为X中的每个示例预测一个类
    
    参数：
        parameters - 包含你的参数的python字典
        X - 输入数据的大小（n_x，m）
    
    返回
        predictions - 我们模型预测的向量（红色：0 /蓝色：1）
     
     """
    A2 , cache = forward_propagation(X,parameters)
    predictions = np.round(A2)
    
    return predictions

#测试predict
print("=========================测试predict=========================")

parameters, X_assess = predict_test_case()

predictions = predict(parameters, X_assess)
print("预测的平均值 = " + str(np.mean(predictions)))


# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h = 4, num_iterations=10000, print_cost=True)

"""
plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50] #隐藏层数量
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print ("隐藏层的节点数量： {}  ，准确率: {} %".format(n_h, accuracy))
    
"""
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

dataset = "noisy_moons"

X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

if dataset == "blobs":
    Y = Y % 2

plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);