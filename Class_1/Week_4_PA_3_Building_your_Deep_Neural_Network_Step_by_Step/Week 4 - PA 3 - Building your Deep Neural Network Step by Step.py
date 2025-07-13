# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 16:32:27 2018

@author: Oscar
"""


"""
在这款笔记本中，您将实现构建深度神经网络所需的所有功能。
在下一个作业中，您将使用这些函数为图像分类构建深度神经网络。
完成这项任务后，您将能够：

使用像ReLU这样的非线性单位来改进你的模型
建立一个更深的神经网络（具有多于一个隐藏层）
实现一个易于使用的神经网络类


"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCases
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
import lr_utils

#%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#%load_ext autoreload
#%autoreload 2

np.random.seed(1)

"""
 - 作业大纲
为了建立你的神经网络，你将实施几个“辅助功能”。
这些辅助函数将在下一个任务中用于构建一个双层神经网络和一个L层神经网络。
您将执行的每个小助手功能都将有详细的说明，以指导您完成必要的步骤。以下是这项任务的概要，您将：

初始化双层网络和LL层神经网络的参数。
实现前向传播模块（在下图中以紫色显示）。
完成图层前向传播步骤的LINEAR部分（生成Z [1] Z [1]）。
我们为您提供ACTIVATION功能（relu / sigmoid）。
将前两个步骤合并为一个新的[LINEAR-> ACTIVATION]转发函数。
堆叠[LINEAR-> RELU]前进函数L-1时间（对于层1到L-1）并在末尾添加[LINEAR-> SIGMOID]（对于最终层LL）。这给你一个新的L_model_forward函数。
计算损失。
实现反向传播模块（在下图中用红色表示）。
完成图层后向传播步骤的LINEAR部分。
我们给你ACTIVATE函数的渐变（relu_backward / sigmoid_backward）
将前两个步骤合并为一个新的[LINEAR-> ACTIVATION]后退功能。
将[LINEAR-> RELU]向后叠加L-1次，并在新的L_model_backward函数中向后添加[LINEAR-> SIGMOID]
最后更新参数。


请注意，对于每个前向函数，都有一个相应的后向函数。 这就是为什么在你的转发模块的每一步你都会在缓存中存储一些值。
缓存的值对计算渐变很有用。 在反向传播模块中，您将使用缓存来计算渐变。 这项任务将向您显示如何执行每个步骤。


3 - 初始化
您将编写两个辅助函数，它们将初始化模型的参数。 第一个函数将用于初始化两层模型的参数。 第二个将这个初始化过程推广到LL层。

3.1 - 2层神经网络
练习：创建并初始化2层神经网络的参数。

说明：

模型的结构是：LINEAR - > RELU - > LINEAR - > SIGMOID。
对权重矩阵使用随机初始化。 使用正确形状的np.random.randn（shape）* 0.01。
对偏差使用零初始化。 使用np.zeros（形状）。
"""
def initialize_parameters(n_x,n_h,n_y):
    """
    参数：
        n_x - 输入层节点数量
        n_h - 隐藏层节点数量
        n_y - 输出层节点数量
    
    返回：
        parameters - 包含你的参数的python字典：
            W1 - 权重矩阵,维度为（n_h，n_x）
            b1 - 偏向量，维度为（n_h，1）
            W2 - 权重矩阵，维度为（n_y，n_h）
            b2 - 偏向量，维度为（n_y，1）

    """
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    
    #使用断言确保我的数据格式是正确的
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters  

#测试initialize_parameters
print("==============测试initialize_parameters==============")
parameters = initialize_parameters(2,2,1)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

"""
3.2 - L层神经网络
更深层的L层神经网络的初始化更复杂，因为有更多的权重矩阵和偏向量。 
在完成initialize_parameters_deep时，应确保每个图层之间的尺寸匹配。 
回想一下，n [l]是层l中的单位数。 
因此，例如，如果我们的输入X的大小是（12288,209）（12288,209）（其中m = 209m = 209的例子），那么：


Shape of W	Shape of b	Activation	Shape of Activation
Layer 1	 (n[1],12288)(n[1],12288) 	 (n[1],1)(n[1],1) 	 Z[1]=W[1]X+b[1]Z[1]=W[1]X+b[1] 	 (n[1],209)(n[1],209) 
Layer 2	 (n[2],n[1])(n[2],n[1]) 	 (n[2],1)(n[2],1) 	 Z[2]=W[2]A[1]+b[2]Z[2]=W[2]A[1]+b[2] 	 (n[2],209)(n[2],209) 
⋮⋮ 	 ⋮⋮ 	 ⋮⋮ 	 ⋮⋮ 	 ⋮⋮ 
Layer L-1	 (n[L−1],n[L−2])(n[L−1],n[L−2]) 	 (n[L−1],1)(n[L−1],1) 	 Z[L−1]=W[L−1]A[L−2]+b[L−1]Z[L−1]=W[L−1]A[L−2]+b[L−1] 	 (n[L−1],209)(n[L−1],209) 
Layer L	 (n[L],n[L−1])(n[L],n[L−1]) 	 (n[L],1)(n[L],1) 	 Z[L]=W[L]A[L−1]+b[L]Z[L]=W[L]A[L−1]+b[L] 	 (n[L],209)(n[L],209) 

请记住，当我们在Python中计算WX + bWX + b时，它会执行广播。 例如，如果：




练习：实现L层神经网络的初始化。

说明：

模型的结构是[LINEAR - > RELU]××（L-1） - > LINEAR - > SIGMOID。即，它具有使用ReLU激活功能的L-1L-1层，接着是具有S形激活功能的输出层。
对权重矩阵使用随机初始化。使用np.random.rand（shape）* 0.01。
对偏见使用零初始化。使用np.zeros（形状）。
我们将n [l] n [l]（不同图层中的单元数）存储在变量layers_dims中。例如，上周“平面数据分类模型”的layers_dims可能是[2,4,1]：有两个输入，一个隐藏层有四个隐藏单元，输出层有一个输出单元。因此，W1的形状为（4,2），b1为（4,1），W2为（1,4），b2为（1,1）。现在您将推广到LL层！
这里是L = 1L = 1（单层神经网络）的实现。它应该激励你实现一般情况（L层神经网络）。
  如果L == 1：
      参数[“W”+ str（L）] = np.random.randn（layers_dims [1]，layers_dims [0]）* 0.01
      参数[“b”+ str（L）] = np.zeros（（layers_dims [1]，1））

"""

def initialize_parameters_deep(layers_dims):
    """
    参数：
        layers_dims - 包含我们网络中每个图层的尺寸的python数组（列表）
    
    返回：
        parameters - 包含参数“W1”，“b1”，...，“WL”，“bL”的python字典：
                     W1 - 形状的权重矩阵（layers_dims [1]，layers_dims [1-1]）
                     bl - 形状的偏向量（layers_dims [1]，1）
    """
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)
    
    for l in range(1,L):
        #parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) / np.sqrt(layers_dims[l - 1])
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layers_dims[l], 1))
        
    return parameters

#测试initialize_parameters_deep
print("==============测试initialize_parameters_deep==============")
layers_dims = [5,4,3]
parameters = initialize_parameters_deep(layers_dims)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

"""
4 - 前向传播模块
4.1 - 线性转发
现在您已初始化参数，您将执行前向传播模块。

 您将开始实施一些基本功能，稍后您将在实施该模型时使用这些功能。
 您将按以下顺序完成三项功能：

LINEAR
LINEAR - >ACTIVATION，其中激活将是ReLU或Sigmoid。
[LINEAR - > RELU] ×（L-1） - > LINEAR - > SIGMOID（整个模型）
线性正向模块（向量化所有示例）计算以下等式：
"""

def linear_forward(A,W,b):
    """
    实现图层前向传播的线性部分。

    参数：
        A - 来自上一层（或输入数据）的激活:(上一层的大小，示例的数量）
        W - 权重矩阵：形状的numpy数组（当前图层的大小，前一图层的大小）
        b - 偏向量，形状的numpy阵列（当前图层的大小，1）

    返回：
         Z - 激活功能的输入，也称为预激活参数
         cache - 一个包含“A”，“W”和“b”的Python字典; 存储以有效地计算后向传递
    """
    Z = np.dot(W,A) + b
    assert(Z.shape == (W.shape[0],A.shape[1]))
    cache = (A,W,b)
     
    return Z,cache

#测试linear_forward
print("==============测试linear_forward==============")
A,W,b = testCases.linear_forward_test_case()
Z,linear_cache = linear_forward(A,W,b)
print("Z = " + str(Z))


"""
为了更方便，您将把两个功能（线性和激活）分组为一个功能（LINEAR-> ACTIVATION）。 
因此，您将实现一个执行LINEAR前进步骤，然后执行ACTIVATION前进步骤的功能。

练习：实现LINEAR-> ACTIVATION图层的向前传播。 
数学关系为：A [1] = g（Z [1]）= g（W [1] A [l-1] + b [1] W [l] A [l-1] + b [l]），其中激活“g”可以是sigmoid（）或relu（）。
 使用linear_forward（）和正确的激活函数。
"""
def linear_activation_forward(A_prev,W,b,activation="sigmoid"):
    """
    实现LINEAR-> ACTIVATION图层的前向传播

    参数：
        A_prev - 来自上一层（或输入数据）的激活:(上一层的大小，示例数）
        W - 权重矩阵：形状的numpy数组（当前图层的大小，前一图层的大小）
        b - 偏向量，形状的numpy阵列（当前图层的大小，1）
        activation - 要在此图层中使用的激活，以文本字符串形式存储：“sigmoid”或“relu”

    返回：
        A - 激活函数的输出，也称为激活后值
        cache - 一个包含“linear_cache”和“activation_cache”的Python字典;存储以有效地计算后向传递
    """
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert(A.shape == (W.shape[0],A_prev.shape[1]))
    cache = (linear_cache,activation_cache)
    
    return A,cache

#测试linear_activation_forward
print("==============测试linear_activation_forward==============")
A_prev, W,b = testCases.linear_activation_forward_test_case()
A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
print("With sigmoid: A = " + str(A))

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
print("With ReLU: A = " + str(A))

"""
在深度学习中，“[LINEAR-> ACTIVATION]”计算在神经网络中被计为单层，而不是两层。

d）L层模型
为了在实现LL层神经网时更加方便，
您需要一个函数来复制前一个函数
（带有RELU的linear_activation_forward）L-1次，
然后用一个带有SIGMOID的linear_activation_forward跟踪它。

"""

def L_model_forward(X,parameters):
    """
    实现[LINEAR-> RELU] *（L-1） - > LINEAR-> SIGMOID计算的前向传播
    
    参数：
        X - 数据，形状的numpy数组（输入大小，示例数）
        parameters - initialize_parameters_deep（）的输出
    
    返回：
        AL - 最后激活值
        caches - 包含以下内容的缓存列表：
                 linear_relu_forward（）的每个缓存（其中有L-1个，索引从0到L-2）
                 linear_sigmoid_forward（）的缓存（有一个索引的L-1）
    """
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1,L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
    
    return AL,caches

#测试L_model_forward
print("==============测试L_model_forward==============")
X,parameters = testCases.L_model_forward_test_case()
AL,caches = L_model_forward(X,parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))

"""
现在您将执行向前和向后传播。 你需要计算成本，因为你想检查你的模型是否真的在学习。
"""

def compute_cost(AL,Y):
    """
    实施等式（7）定义的成本函数。

    参数：
        AL - 与标签预测相对应的概率向量，形状（1，示例数量）
        Y - 真正的“标签”向量（例如：如果非cat，则包含0，如果cat为1），shape（1，示例数）

    返回：
        cost - 交叉熵成本
    """
    m = Y.shape[1]
    cost = -np.sum(np.multiply(np.log(AL),Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m
        
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return cost

#测试compute_cost
print("==============测试compute_cost==============")
Y,AL = testCases.compute_cost_test_case()
print("cost = " + str(compute_cost(AL, Y)))


"""
就像向前传播一样，您将实施用于反向传播的辅助函数。 
请记住，反向传播用于计算相对于参数的损失函数的梯度。
"""

def linear_backward(dZ,cache):
    """
    为单层实现反向传播的线性部分（第L层）

    参数：
         dZ - 相对于（当前第l层的）线性输出的成本梯度
         cache - 来自当前图层前向传播的值的元组（A_prev，W，b）

    返回：
         dA_prev - 相对于激活（前一层l-1）的成本梯度，与A_prev形状相同
         dW - 相对于W（当前层l）的成本梯度，与W的形状相同
         db - 相对于b（当前层l）的成本梯度，与b形状相同
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db
#测试linear_backward
print("==============测试linear_backward==============")
dZ, linear_cache = testCases.linear_backward_test_case()

dA_prev, dW, db = linear_backward(dZ, linear_cache)
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))


"""
接下来，我们需要计算激活函数的反向传播函数：
"""

def linear_activation_backward(dA,cache,activation="relu"):
    """
    实现LINEAR-> ACTIVATION图层的后向传播。
    
    参数：
         dA - 当前层l的激活后梯度
         cache - 我们存储的用于有效计算反向传播的值的元组（值为linear_cache，activation_cache）
         activation - 要在此图层中使用的激活，以文本字符串形式存储：“sigmoid”或“relu”
    
    返回：
         dA_prev - 相对于激活（前一层l-1）的成本梯度，与A_prev形状相同
         dW - 相对于W（当前层l）的成本梯度，与W的形状相同
         db - 相对于b（当前层l）的成本梯度，与b形状相同
    """
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev,dW,db

#测试linear_activation_backward
print("==============测试linear_activation_backward==============")
AL, linear_activation_cache = testCases.linear_activation_backward_test_case()
 
dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
print ("sigmoid:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db) + "\n")
 
dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
print ("relu:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))

"""
6.3 - L模型向后
现在您将实现整个网络的后向功能。
回想一下，当您实现L_model_forward函数时，在每次迭代中，您都存储了一个包含（X，W，b和z）的缓存。
在反向传播模块中，您将使用这些变量来计算梯度。因此，在L_model_backward函数中，您将从层LL开始向后遍历所有隐藏层。
在每一步中，您将使用层ll的缓存值通过层ll反向传播。下面的图5显示了反向传球。



图5：向后传球
初始化反向传播：为了通过这个网络反向传播，我们知道输出是A [L] =σ（Z [L]）A [L] =σ（Z [L]）。
因此你的代码需要计算dAL =∂∂A[L] =∂L∂A[L]。为此，请使用此公式（使用不需要深入了解的微积分）：

dAL = - （np.divide（Y，AL） - np.divide（1 - Y，1 - AL））＃对于AL的成本导数
然后，您可以使用此激活后渐变dAL继续向后。
如图5所示，现在可以将dAL输入到您实现的LINEAR-> SIGMOID后向函数中（它将使用由L_model_forward函数存储的缓存值）。
之后，您将不得不使用for循环使用LINEAR-> RELU后向函数遍历所有其他图层。
您应该将每个dA，dW和db存储在grads字典中。为此，请使用以下公式：

梯度[ “DW” + STR（升）] =一页[1]（15）
（15）梯度[ “DW” + STR（升）] =一页[1]
 
例如，对于l = 3l = 3，这将在梯度[“dW3”]中存储dW [l] dW [l]。

练习：实现[LINEAR-> RELU]××（L-1） - > LINEAR - > SIGMOID模型的反向传播。
"""
def L_model_backward(AL,Y,caches):
    """
    对[LINEAR-> RELU] *（L-1） - > LINEAR - > SIGMOID组执行反向传播
    
    参数：
     AL - 概率向量，正向传播的输出（L_model_forward（））
     Y - 真正的“标签”向量（如果非cat，则包含0，如果cat为1，则包含）
     caches - 包含以下内容的缓存列表：
                 每个具有“relu”的linear_activation_forward（）缓存（它是缓存[l]，对于范围（L-1）中的l，即l = 0 ... L-2）
                 linear_activation_forward（）与“sigmoid”（缓存[L-1]）的缓存
    
    返回：
     grads - 具有渐变的字典
              grads [“dA”+ str（l）] = ...
              grads [“dW”+ str（l）] = ...
              grads [“db”+ str（l）] = ...
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    
    return grads
#测试L_model_backward
print("==============测试L_model_backward==============")
AL, Y_assess, caches = testCases.L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dA1 = "+ str(grads["dA1"]))

"""
6.4 - 更新参数
在本节中，您将使用渐变下降更新模型的参数：

W [1] = W [1]-αdW [1]（16）
（16）W [1] = W [1]-αdW [1]
 
b [1] = b [1]-αdb [1]（17）
（17）b [1] = b [1]-αdb [1]
 
其中αα是学习率。 计算更新后的参数后，将它们存储在参数字典中。

练习：使用渐变下降实现update_parameters（）以更新参数。

说明：对于l = 1,2，...，Ll = 1,2，...，L，在每个W [1] W [1]和b [1] b [1]上使用梯度下降来更新参数。

"""

def update_parameters(parameters, grads, learning_rate):
    """
    使用渐变下降更新参数
    
    参数：
     parameters - 包含你的参数的python字典
     grads - 包含渐变的python字典，L_model_backward的输出
    
    返回：
     parameters - 包含更新参数的python字典
                   参数[“W”+ str（l）] = ...
                   参数[“b”+ str（l）] = ...
    """
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        
    return parameters

#测试update_parameters
print("==============测试update_parameters==============")
parameters, grads = testCases.update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)
 
print ("W1 = "+ str(parameters["W1"]))
print ("b1 = "+ str(parameters["b1"]))
print ("W2 = "+ str(parameters["W2"]))
print ("b2 = "+ str(parameters["b2"]))


"""
至此为止，我们已经实现该神经网络中，所有需要的函数。

接下来，我们将这些方法组合在一起，构成一个神经网络类，可以方便的使用

您将使用您在之前的作业中实施的功能来构建深度网络，并将其应用于猫与非猫分类。 
希望相对于之前的逻辑回归实现，您将看到精度的提高。

完成这项任务后，您将能够：

建立并应用深度神经网络来监督学习。
让我们开始吧！


2 - 数据集
您将使用与“Logistic回归作为神经网络”相同的“Cat vs non-Cat”数据集（作业2）。 
您建立的模型在分类猫与非猫图像时的测试精度为70％。 希望你的新模型能更好地发挥作用！

问题陈述：给你一个数据集（“data.h5”），其中包含：

- 标记为cat（1）或非cat（0）的训练集的m_train图像，
- 一组m_test图像，标记为猫和非猫
- 每个图像的形状（num_px，num_px，3），其中3是3个通道（RGB）。
让我们更熟悉数据集。 运行下面的单元格加载数据。
"""


train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = lr_utils.load_dataset()

m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]

print ("训练集的数量: m_train = " + str(m_train))
print ("测试集的数量 : m_test = " + str(m_test))
print ("每张图片的宽/高 : num_px = " + str(num_px))
print ("每张图片的大小 : (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("训练集_图片的维数 : " + str(train_set_x_orig.shape))
print ("训练集_标签的维数 : " + str(train_set_y.shape))
print ("测试集_图片的维数: " + str(test_set_x_orig.shape))
print ("测试集_标签的维数: " + str(test_set_y.shape))


train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

"""
现在您已熟悉数据集，现在可以构建一个深层神经网络来区分猫图像和非猫图像。

您将构建两种不同的模型：

一个2层神经网络
一个L层深度神经网络
然后，您将比较这些模型的性能，并尝试使用不同的L值。

我们来看看这两种架构。
"""

"""
4-两层神经网络
问题：使用你在前面任务中实现的辅助函数来构建一个具有以下结构的2层神经网络：LINEAR - > RELU - > LINEAR - > SIGMOID。 
您可能需要的功能及其输入是：

def initialize_parameters(n_x, n_h, n_y):
    ...
    return parameters 
def linear_activation_forward(A_prev, W, b, activation):
    ...
    return A, cache
def compute_cost(AL, Y):
    ...
    return cost
def linear_activation_backward(dA, cache, activation):
    ...
    return dA_prev, dW, db
def update_parameters(parameters, grads, learning_rate):
    ...
    return parameters
    
"""

n_x = 12288
n_h = 7
n_y = 1
layers_dims = (n_x,n_h,n_y)


def two_layer_model(X,Y,layers_dims,learning_rate=0.0075,num_iterations=3000,print_cost=False,isPlot=True):
    """
    实现一个两层的神经网络，【LINEAR->RELU】 -> 【LINEAR->SIGMOID】
    参数：
        X - 输入的数据，维度为(n_x，例子数)
        Y - 标签，向量，0为非猫，1为猫，维度为(1,数量)
        layers_dims - 层数的向量，维度为(n_y,n_h,n_y)
        learning_rate - 学习率
        num_iterations - 迭代的次数
        print_cost - 是否打印成本值，每100次打印一次
    返回:
        parameters - 一个包含W1，b1，W2，b2的字典变量
    """
    np.random.seed(1)
    grads = {}
    costs = []
    (n_x,n_h,n_y) = layers_dims
    
    """
    初始化参数
    """
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    """
    开始进行迭代
    """
    for i in range(0,num_iterations):
        #前向传播
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
        
        #计算成本
        cost = compute_cost(A2,Y)
        
        #后向传播
        ##初始化后向传播
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        ##向后传播，输入：“dA2，cache2，cache1”。 输出：“dA1，dW2，db2;还有dA0（未使用），dW1，db1”。
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
        
        ##向后传播完成后的数据保存到grads
        grads["dW1"] = dW1
        grads["db1"] = db1
        grads["dW2"] = dW2
        grads["db2"] = db2
        
        #更新参数
        parameters = update_parameters(parameters,grads,learning_rate)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        #打印成本值，如果print_cost=False则忽略
        if i % 100 == 0:
            #记录成本
            costs.append(cost)
            #是否打印成本值
            if print_cost:
                print("第", i ,"次迭代，成本值为：" ,np.squeeze(cost))
    #迭代完成，根据条件绘制图
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    
    #返回parameters
    return parameters


#parameters = two_layer_model(train_x, train_set_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True,isPlot=True)


"""
现在已经训练完成，开始训练
"""
def predict(X, y, parameters):
    """
    该函数用于预测L层神经网络的结果。
    
    参数：
     X - 您想要标记的示例数据集
     y - 标签
     parameters - 训练模型的参数
    
    返回：
     p - 给定数据集X的预测
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)
 
    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    print("Accuracy: "  + str(float(np.sum((p == y))/m)))
        
    return p

#predictions_train = predict(train_x, train_set_y, parameters)
#predictions_test = predict(test_x, test_set_y, parameters)

"""
注意：您可能会注意到以较少的迭代运行模型（比如1500）可以提高测试集的准确性。 
这被称为“早期停止”，我们将在下一个课程中讨论它。 提前停止是防止过度配合的一种方法。

恭喜！ 看来你的2层神经网络比逻辑回归实现（70％，第2周）具有更好的性能（72％）。
 让我们来看看你是否能用LL层模型做得更好。
"""

"""
我们来构建多层神经网络
"""
# layers_dims = [12288, 20, 7, 5, 1] #  5层模型
def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False,isPlot=True): #lr was 0.009
    """
    实现一个L层神经网络：[LINEAR-> RELU] *（L-1） - > LINEAR-> SIGMOID。
    
    参数：
     X - 数据，形状的numpy数组（示例数，num_px * num_px * 3）
     Y - 真正的“标签”向量（包含0，如果是猫，则为1，如果非猫），形状（1，示例数）
     layers_dims - 包含输入大小和每个图层大小，长度（层数+ 1）的列表。
     learning_rate - 梯度下降更新规则的学习率
     num_iterations - 优化循环的迭代次数
     print_cost - 如果为True，则每100步打印一次成本
    
    返回：
     parameters - 模型学习的参数。 然后他们可以用来预测。
    """
    np.random.seed(1)
    costs = []
    
    parameters = initialize_parameters_deep(layers_dims)
    
    for i in range(0,num_iterations):
        AL , caches = L_model_forward(X,parameters)
        
        cost = compute_cost(AL,Y)
        
        grads = L_model_backward(AL,Y,caches)
        
        parameters = update_parameters(parameters,grads,learning_rate)
        
        #打印成本值，如果print_cost=False则忽略
        if i % 100 == 0:
            #记录成本
            costs.append(cost)
            #是否打印成本值
            if print_cost:
                print("第", i ,"次迭代，成本值为：" ,np.squeeze(cost))
    #迭代完成，根据条件绘制图
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    return parameters

"""
您现在将模型作为5层神经网络进行训练。

运行下面的单元格来训练你的模型。 
每次迭代都应该降低成本。
 运行2500次迭代可能需要长达5分钟的时间。 
 检查“迭代0后的成本”是否与下面的预期输出匹配，如果没有单击方块（⬛）停止单元格并尝试查找错误。
"""

parameters = L_layer_model(train_x, train_set_y, layers_dims, num_iterations=2500, print_cost=True,isPlot=True)

pred_train = predict(train_x, train_set_y, parameters)

pred_test = predict(test_x, test_set_y, parameters)






