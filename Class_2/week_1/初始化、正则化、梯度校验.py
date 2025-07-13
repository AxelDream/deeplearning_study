# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 09:34:44 2018

@author: Oscar
"""

"""
初始化

欢迎来到“改进深神经网络”的第一个任务。

训练神经网络需要指定权值的初始值。精心挑选的初始化方法有助于学习。

如果您完成了这个专业化的前一步，您可能会按照我们的说明进行权重初始化，而且到目前为止已经完成了。但是如何选择一个新的神经网络的初始化呢？在这个笔记本上，你会看到不同的初始化，导致不同的结果。

选择良好的初始化可以：

加快梯度下降的收敛速度

增加梯度下降的概率收敛到较低的训练（和泛化）错误。

要启动，请运行下面的单元格来加载包和将要进行分类的平面数据集。
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils #第一部分，初始化
import reg_utils #第二部分，正则化
import gc_utils #第三部分，梯度检查
#%matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

'''

# load image dataset: blue/red dots in circles
train_X, train_Y, test_X, test_Y = init_utils.load_dataset()

#我们将要建立一个分类器把蓝点和红点分开。

"""
在之前我们已经实现过一个3层的神经网络，我们将对它进行初始化：

我们将会试着下面几种初始化方法:
    初始化为0：在输入参数中全部初始化为0，参数名为initialization = "zeros"，核心代码：
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
    初始化为随机数：把输入参数设置为随机值，权重初始化为大的随机值。参数名为initialization = "random"，核心代码：
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10
    抑梯度异常初始化：参见梯度消失和梯度爆炸的那一个视频，参数名为initialization = "he"，核心代码：
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
        
"""

def model(X,Y,learning_rate=0.01,num_iterations=15000,print_cost=True,initialization="he",is_polt=True):
    """
    实现一个三层的神经网络：LINEAR ->RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    
    参数：
        X - 输入的数据，维度为(2, 要训练/测试的数量)
        Y - 标签，【0 | 1】，维度为(1，对应的是输入的数据的标签)
        learning_rate - 学习速率
        num_iterations - 迭代的次数
        print_cost - 是否打印成本值，每迭代1000次打印一次
        initialization - 字符串类型，初始化的类型【"zeros" | "random" | "he"】
        is_polt - 是否绘制梯度下降的曲线图
    返回
        parameters - 学习后的参数
    """
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0],10,5,1]
    
    #选择初始化参数的类型
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)
    else : 
        print("错误的初始化参数！程序退出")
        exit
    
    #开始学习
    for i in range(0,num_iterations):
        #前向传播
        a3 , cache = init_utils.forward_propagation(X,parameters)
        
        #计算成本        
        cost = init_utils.compute_loss(a3,Y)
        
        #反向传播
        grads = init_utils.backward_propagation(X,Y,cache)
        
        #更新参数
        parameters = init_utils.update_parameters(parameters,grads,learning_rate)
        
        #记录成本
        if i % 1000 == 0:
            costs.append(cost)
            #打印成本
            if print_cost:
                print("第" + str(i) + "次迭代，成本值为：" + str(cost))
        
    
    #学习完毕，绘制成本曲线
    if is_polt:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    
    #返回学习完毕后的参数
    return parameters

"""
0初始化，我们需要对w和b进行初始化
"""
def initialize_parameters_zeros(layers_dims):
    """
    将模型的参数全部设置为0
    
    参数：
        layers_dims - 列表，模型的层数和对应每一层的节点的数量
    返回
        parameters - 包含了所有W和b的字典
            W1 - 权重矩阵，维度为（layers_dims[1], layers_dims[0]）
            b1 - 偏置向量，维度为（layers_dims[1],1）
            ···
            WL - 权重矩阵，维度为（layers_dims[L], layers_dims[L -1]）
            b1 - 偏置向量，维度为（layers_dims[L],1）
    """
    parameters = {}
    
    L = len(layers_dims) #网络层数
    
    for l in range(1,L):
        parameters["W" + str(l)] = np.zeros((layers_dims[l],layers_dims[l-1]))
        parameters["b" + str(l)] = np.zeros((layers_dims[l],1))
        
        #使用断言确保我的数据格式是正确的
        assert(parameters["W" + str(l)].shape == (layers_dims[l],layers_dims[l-1]))
        assert(parameters["b" + str(l)].shape == (layers_dims[l],1))
        
    return parameters

"""
parameters = initialize_parameters_zeros([3,2,1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

W1 = [[ 0.  0.  0.]
 [ 0.  0.  0.]]
b1 = [[ 0.]
 [ 0.]]
W2 = [[ 0.  0.]]
b2 = [[ 0.]]
"""

"""

parameters = model(train_X, train_Y, initialization = "zeros",is_polt=False)
print ("训练集:")
predictions_train = init_utils.predict(train_X, train_Y, parameters)
print ("测试集:")
predictions_test = init_utils.predict(test_X, test_Y, parameters)

"""


"""

第0次迭代，成本值为：0.69314718056
第1000次迭代，成本值为：0.69314718056
第2000次迭代，成本值为：0.69314718056
第3000次迭代，成本值为：0.69314718056
第4000次迭代，成本值为：0.69314718056
第5000次迭代，成本值为：0.69314718056
第6000次迭代，成本值为：0.69314718056
第7000次迭代，成本值为：0.69314718056
第8000次迭代，成本值为：0.69314718056
第9000次迭代，成本值为：0.69314718056
第10000次迭代，成本值为：0.69314718056
第11000次迭代，成本值为：0.69314718056
第12000次迭代，成本值为：0.69314718056
第13000次迭代，成本值为：0.69314718056
第14000次迭代，成本值为：0.69314718056
训练集:
Accuracy: 0.5
测试集:
Accuracy: 0.5
"""

"""
性能确实很差，而且成本并没有真正降低，算法的性能也比随机猜测要好。为什么？让我们看看预测和决策边界的细节：
"""
"""
print("predictions_train = " + str(predictions_train))
print("predictions_test = " + str(predictions_test))

plt.title("Model with Zeros initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)
"""

"""
predictions_train = [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0]]
predictions_test = [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
"""

"""
该模型预测每个示例的0。

通常来说，零初始化都会导致神经网络无法打破对称性，最终导致的结构就是无论网络有多少层，最终只能得到和Logistic函数相同的效果。
"""

"""
3 -随机初始化

为了打破对称，我们可以随机地把权值化。
在随机初始化之后，每个神经元可以开始学习其输入的不同功能。
在这个练习中，你将会看到如果权重是随机的，但是对于非常大的值会发生什么。


练习:执行下列函数，将你的权重初始化为大的随机值(按*10按比例缩放)，你的偏差值为零。
使用np.random.randn(.. ..) * 10表示权重和np. 0。偏见,. .))。
我们使用一个固定的np.random.seed(..)来确保您的“随机”权重与我们的匹配，因此，如果运行几次您的代码将始终为参数提供相同的初始值，请不要担心。
"""

def initialize_parameters_random(layers_dims):
    """
    参数：
        layers_dims - 列表，模型的层数和对应每一层的节点的数量
    返回
        parameters - 包含了所有W和b的字典
            W1 - 权重矩阵，维度为（layers_dims[1], layers_dims[0]）
            b1 - 偏置向量，维度为（layers_dims[1],1）
            ···
            WL - 权重矩阵，维度为（layers_dims[L], layers_dims[L -1]）
            b1 - 偏置向量，维度为（layers_dims[L],1）
    """
    
    np.random.seed(3)               # This seed makes sure your "random" numbers will be the as ours
    parameters = {}
    L = len(layers_dims)            # integer representing the number of layers
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        
        #使用断言确保我的数据格式是正确的
        assert(parameters["W" + str(l)].shape == (layers_dims[l],layers_dims[l-1]))
        assert(parameters["b" + str(l)].shape == (layers_dims[l],1))
        
    return parameters

"""
parameters = initialize_parameters_random([3, 2, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))



W1 = [[ 17.88628473   4.36509851   0.96497468]
 [-18.63492703  -2.77388203  -3.54758979]]
b1 = [[ 0.]
 [ 0.]]
W2 = [[-0.82741481 -6.27000677]]
b2 = [[ 0.]]
"""
"""
parameters = model(train_X, train_Y, initialization = "random",is_polt=True)
print("训练集：")
predictions_train = init_utils.predict(train_X, train_Y, parameters)
print("测试集：")
predictions_test = init_utils.predict(test_X, test_Y, parameters)

print(predictions_train)
print(predictions_test)


plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)

"""


"""
第0次迭代，成本值为：inf
第1000次迭代，成本值为：0.625098279396
第2000次迭代，成本值为：0.59812165967
第3000次迭代，成本值为：0.56384175723
第4000次迭代，成本值为：0.55017030492
第5000次迭代，成本值为：0.544463290966
第6000次迭代，成本值为：0.5374513807
第7000次迭代，成本值为：0.476404207407
第8000次迭代，成本值为：0.397814922951
第9000次迭代，成本值为：0.393476402877
第10000次迭代，成本值为：0.392029546188
第11000次迭代，成本值为：0.389245981351
第12000次迭代，成本值为：0.386154748571
第13000次迭代，成本值为：0.38498472891
第14000次迭代，成本值为：0.382782830835
训练集：
Accuracy: 0.83
测试集：
Accuracy: 0.86
[[1 0 1 1 0 0 1 1 1 1 1 0 1 0 0 1 0 1 1 0 0 0 1 0 1 1 1 1 1 1 0 1 1 0 0 1 1
  1 1 1 1 1 1 0 1 1 1 1 0 1 0 1 1 1 1 0 0 1 1 1 1 0 1 1 0 1 0 1 1 1 1 0 0 0
  0 0 1 0 1 0 1 1 1 0 0 1 1 1 1 1 1 0 0 1 1 1 0 1 1 0 1 0 1 1 0 1 1 0 1 0 1
  1 0 0 1 0 0 1 1 0 1 1 1 0 1 0 0 1 0 1 1 1 1 1 1 1 0 1 1 0 0 1 1 0 0 0 1 0
  1 0 1 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1 0 1 1 0 1 0 1 1 1 1 0 1 1 1 1 0 1 0 1
  0 1 1 1 1 0 1 1 0 1 1 0 1 1 0 1 0 1 1 1 0 1 1 1 0 1 0 1 0 0 1 0 1 1 0 1 1
  0 1 1 0 1 1 1 0 1 1 1 1 0 1 0 0 1 1 0 1 1 1 0 0 0 1 1 0 1 1 1 1 0 1 1 0 1
  1 1 0 0 1 0 0 0 1 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 1 1 1 1 1 1 1 0 0 0 1
  1 1 1 0]]
[[1 1 1 1 0 1 0 1 1 0 1 1 1 0 0 0 0 1 0 1 0 0 1 0 1 0 1 1 1 1 1 0 0 0 0 1 0
  1 1 0 0 1 1 1 1 1 0 1 1 1 0 1 0 1 1 0 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 0 1 1
  1 1 1 0 1 0 0 1 0 0 0 1 1 0 1 1 0 0 0 1 1 0 1 1 0 0]]
"""

"""
观察:


成本开始很高。这是因为由于具有较大的随机值权重，最后一个激活(sigmoid)输出的结果非常接近于0或1，而当它出现错误时，它会导致非常高的损失。
实际上,当日志([3])=日志(0)日志⁡([3])=日志⁡(0)损失趋于无穷。

糟糕的初始化会导致消失/爆炸梯度，这也会减慢优化算法。

如果您对这个网络进行更长时间的训练，您将看到更好的结果，但是使用过大的随机数初始化会减慢优化的速度。


总而言之:


将权重初始化为非常大的随机值并不能很好地工作。

希望小随机值的初始化效果更好。重要的问题是:这些随机值应该有多小?让我们在下一部分找到答案!

"""

def initialize_parameters_he(layers_dims):
    """
    参数：
        layers_dims - 列表，模型的层数和对应每一层的节点的数量
    返回
        parameters - 包含了所有W和b的字典
            W1 - 权重矩阵，维度为（layers_dims[1], layers_dims[0]）
            b1 - 偏置向量，维度为（layers_dims[1],1）
            ···
            WL - 权重矩阵，维度为（layers_dims[L], layers_dims[L -1]）
            b1 - 偏置向量，维度为（layers_dims[L],1）
    """
    
    np.random.seed(3)               # This seed makes sure your "random" numbers will be the as ours
    parameters = {}
    L = len(layers_dims)            # integer representing the number of layers
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        
        #使用断言确保我的数据格式是正确的
        assert(parameters["W" + str(l)].shape == (layers_dims[l],layers_dims[l-1]))
        assert(parameters["b" + str(l)].shape == (layers_dims[l],1))
        
    return parameters

"""

parameters = initialize_parameters_he([2, 4, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

W1 = [[ 1.78862847  0.43650985]
 [ 0.09649747 -1.8634927 ]
 [-0.2773882  -0.35475898]
 [-0.08274148 -0.62700068]]
b1 = [[ 0.]
 [ 0.]
 [ 0.]
 [ 0.]]
W2 = [[-0.03098412 -0.33744411 -0.92904268  0.62552248]]
b2 = [[ 0.]]

"""

parameters = model(train_X, train_Y, initialization = "he",is_polt=False)
print("训练集:")
predictions_train = init_utils.predict(train_X, train_Y, parameters)
print("测试集:")
init_utils.predictions_test = init_utils.predict(test_X, test_Y, parameters)

"""
第0次迭代，成本值为：0.883053746342
第1000次迭代，成本值为：0.687982591973
第2000次迭代，成本值为：0.675128626452
第3000次迭代，成本值为：0.652611776889
第4000次迭代，成本值为：0.608295897057
第5000次迭代，成本值为：0.530494449172
第6000次迭代，成本值为：0.413864581707
第7000次迭代，成本值为：0.311780346484
第8000次迭代，成本值为：0.236962153303
第9000次迭代，成本值为：0.185972872092
第10000次迭代，成本值为：0.150155562804
第11000次迭代，成本值为：0.123250792923
第12000次迭代，成本值为：0.0991774654653
第13000次迭代，成本值为：0.0845705595402
第14000次迭代，成本值为：0.0735789596268
训练集:
Accuracy: 0.993333333333
测试集:
Accuracy: 0.96
"""

plt.title("Model with He initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)

"""
初始化的模型将蓝色和红色的点在少量的迭代中很好地分离出来。
总结一下：

1. 不同的初始化方法可能导致最终不同的性能

2. 随机初始化有助于打破对称，使得不同隐藏层的单元可以学习到不同的参数。

3. 初始化时，初始值不宜过大。

4. He初始化搭配ReLU激活函数常常可以得到不错的效果。
"""


"""
正则化
在深度学习中，如果数据集没有足够大的话，可能会导致一些过拟合的问题。

过拟合导致的结果就是在训练集上有着很高的精确度，但是在遇到新的样本时，精确度下降严重。

为了避免过拟合的问题，接下来我们要讲解的方式就是正则化。
"""


"""
正则
欢迎来到本周的第二个任务。 如果训练数据集不够大，深度学习模型具有非常大的灵活性和容量，以至于过度拟合可能是一个严重的问题。 

确定它在训练集上表现良好，但是学到的网络并没有推广到它从未见过的新例子！

您将学习：在深度学习模型中使用正则化。

我们首先导入您要使用的软件包。
"""

'''

"""#-----------------------------------------------------------正则化-----------------------------------------------------------#"""
'''

"""
你刚刚被法国足球公司聘为AI专家。 他们希望你推荐法国守门员踢球的位置，这样法国队的球员就可以用他们的头部击球。
假设你现在是一个AI专家，你需要设计一个模型，可以用于推荐在足球场中守门员将球发至哪个位置可以让本队的球员抢到球的可能性更大。

守门员将球踢向空中，每支球队的球员都在努力用头部击球
"""

train_X, train_Y, test_X, test_Y = reg_utils.load_2D_dataset(is_plot=False)

"""
每个点对应于足球场上的一个位置，在法国守门员从足球场左侧射球之后，足球运动员用他/她的头部击球。

如果这个点是蓝色的，这意味着法国球员设法用他/她的头击球
如果这个点是红色的，这意味着其他队员的球员用头撞球
你的目标：使用深度学习模式来找到守门员应在场上踢球的位置。

数据集分析：这个数据集有点吵，但它看起来像是一条将左上半部分（蓝色）与右下半部分（红色）分开的对角线，效果很好。

你将首先尝试一个非正则化模型。 然后，您将学习如何正规化，并决定选择哪种模式来解决法国足球公司的问题。

每一个点对应一个足球落下的位置。

对于蓝色的点，表示我方足球队员抢到球；对于红色的点，则表示对方球员抢到球。

我们的目标是建立一个模型，来找到适合我方球员能抢到球的位置。


- 非正规化模型
您将使用以下神经网络（以下已为您实施）。 这个模型可以使用：

在正则化模式下 - 通过将lambd输入设置为非零值。 我们使用“lambd”而不是“lambda”，因为“lambda”是Python中的保留关键字。
在丢失模式下 - 通过将keep_prob设置为小于1的值
您将首先尝试没有任何正规化的模型。 然后，你将执行：

L2正则化 - 函数：“compute_cost_with_regularization（）”和“backward_propagation_with_regularization（）”
Dropout - 函数：“forward_propagation_with_dropout（）”和“backward_propagation_with_dropout（）”
在每个部分中，您将使用正确的输入运行该模型，以便调用您实施的功能。 看看下面的代码来熟悉模型。
"""

def model(X,Y,learning_rate=0.3,num_iterations=30000,print_cost=True,is_plot=True,lambd=0,keep_prob=1):
    """
    实现一个三层的神经网络：LINEAR ->RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    
    参数：
        X - 输入的数据，维度为(2, 要训练/测试的数量)
        Y - 标签，【0(蓝色) | 1(红色)】，维度为(1，对应的是输入的数据的标签)
        learning_rate - 学习速率
        num_iterations - 迭代的次数
        print_cost - 是否打印成本值，每迭代10000次打印一次，但是每1000次记录一个成本值
        is_polt - 是否绘制梯度下降的曲线图
        lambd - 正则化的超参数，实数
        keep_prob - 
    返回
        parameters - 学习后的参数
    """
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0],20,3,1]
    
    #初始化参数
    parameters = reg_utils.initialize_parameters(layers_dims)
    
    #开始学习
    for i in range(0,num_iterations):
        #前向传播
        ##是否随机删除节点
        if keep_prob == 1:
            ###不随机删除节点
            a3 , cache = reg_utils.forward_propagation(X,parameters)
        elif keep_prob < 1:
            ###随机删除节点
            a3 , cache = forward_propagation_with_dropout(X,parameters,keep_prob)
        else:
            print("keep_prob参数错误！程序退出。")
            exit
        
        #计算成本
        ## 是否使用二范数
        if lambd == 0:
            ###不使用L2正则化
            cost = reg_utils.compute_cost(a3,Y)
        else:
            ###使用L2正则化
            cost = compute_cost_with_regularization(a3,Y,parameters,lambd)
        
        #反向传播
        ##可以同时使用L2正则化和随机删除节点，但是本次实验不同时使用。
        assert(lambd == 0  or keep_prob ==1)
        
        ##两个参数的使用情况
        if (lambd == 0 and keep_prob == 1):
            ### 不使用L2正则化和不使用随机删除节点
            grads = reg_utils.backward_propagation(X,Y,cache)
        elif lambd != 0:
            ### 使用L2正则化，不使用随机删除节点
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            ### 使用随机删除节点，不使用L2正则化
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
        
        #更新参数
        parameters = reg_utils.update_parameters(parameters, grads, learning_rate)
        
        #记录并打印成本
        if i % 1000 == 0:
            ## 记录成本
            costs.append(cost)
            if (print_cost and i % 10000 == 0):
                #打印成本
                print("第" + str(i) + "次迭代，成本值为：" + str(cost))
        
    #是否绘制成本曲线图
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (x1,000)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    
    #返回学习后的参数
    return parameters

"""
parameters = model(train_X, train_Y,is_plot=False)
print("训练集:")
predictions_train = reg_utils.predict(train_X, train_Y, parameters)
print("测试集:")
predictions_test = reg_utils.predict(test_X, test_Y, parameters)

第0次迭代，成本值为：0.655741252348
第10000次迭代，成本值为：0.163299875257
第20000次迭代，成本值为：0.138516424233

我们可以看到，对于训练集，精确度为94%；而对于测试集，精确度为91.5%。

接下来，我们将分割曲线画出来：

plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)

从图中可以看出，在无正则化时，分割曲线有了明显的过拟合特性。
"""


def compute_cost_with_regularization(A3,Y,parameters,lambd):
    """
    实现公式2的L2正则化计算成本
    
    参数：
        A3 - 正向传播的输出结果，维度为（输出节点数量，训练/测试的数量）
        Y - 标签向量，与数据一一对应，维度为(输出节点数量，训练/测试的数量)
        parameters - 包含模型学习后的参数的字典
    返回：
        cost - 使用公式2计算出来的正则化损失的值
    
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    cross_entropy_cost = reg_utils.compute_cost(A3,Y)
    
    L2_regularization_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2))  + np.sum(np.square(W3))) / (2 * m)
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost

"""
当然，因为你改变了成本，你也必须改变向后传播！ 所有的梯度都必须根据这个新的成本来计算。
"""
def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    实现我们添加了L2正则化的基线模型的后向传播。
    
    参数：
        X - 输入数据集，形状（输入大小，示例数量）
        Y - 形状的“真实”标签矢量（输出大小，示例数量）
        cache - 来自forward_propagation（）的cache输出
        lambda - regularization超参数，标量
    
    返回：
        gradients - 一个关于每个参数，激活和预激活变量的渐变字典
    """
    
    m = X.shape[1]
    
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    
    dW3 = (1 / m) * np.dot(dZ3,A2.T) + ((lambd * W3) / m )
    db3 = (1 / m) * np.sum(dZ3,axis=1,keepdims=True)
    
    dA2 = np.dot(W3.T,dZ3)
    dZ2 = np.multiply(dA2,np.int64(A2 > 0))
    dW2 = (1 / m) * np.dot(dZ2,A1.T) + ((lambd * W2) / m)
    db2 = (1 / m) * np.sum(dZ2,axis=1,keepdims=True)
    
    dA1 = np.dot(W2.T,dZ2)
    dZ1 = np.multiply(dA1,np.int64(A1 > 0))
    dW1 = (1 / m) * np.dot(dZ1,X.T) + ((lambd * W1) / m)
    db1 = (1 / m) * np.sum(dZ1,axis=1,keepdims=True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients

'''
'''
parameters = model(train_X, train_Y, lambd=0.7,is_plot=False)
print("使用正则化，训练集:")
predictions_train = reg_utils.predict(train_X, train_Y, parameters)
print("使用正则化，测试集:")
predictions_test = reg_utils.predict(test_X, test_Y, parameters)

"""
使用正则化，训练集:
Accuracy: 0.938388625592
使用正则化，测试集:
Accuracy: 0.93
"""
plt.title("Model with L2-regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)
'''
"""
λ的值是可以使用开发集调整的超参数。
L2正则化使您的决策边界更加平滑。如果λ太大，也可能会“过度平滑”，从而导致具有高偏差的模型。
什么是L2正则化实际上在做什么？

L2正则化依赖于这样的假设，即具有较小权重的模型比具有较大权重的模型更简单。

因此，通过惩罚成本函数中权重的平方值，可以将所有权重驱动到较小的值。拥

有大重量的成本太贵了！这导致更平滑的模型，其中输入变化时输出变化更慢。

你应该记住 -  L2正则化对以下内容的影响：

成本计算：
正规化术语被添加到成本中
反向传播功能：
在权重矩阵方面，渐变中有额外的术语
重量变小（“重量衰减”）：
权重被推到较小的值。
"""

'''
"""
最后，我们使用Dropout来进行正则化，Dropout的原理就是每次迭代过程中随机将其中的一些神经元失效。
在每一次迭代中，你关闭（=设置为零）一个图层的每个神经元，概率为1-keep_prob，或者保持概率为keep_prob（这里为50％）。
丢弃的神经元对迭代的前向和后向传播都无助于训练。

当你关闭一些神经元时，你实际上修改了你的模型。退出背后的想法是，在每次迭代时，您都会训练一个只使用一部分神经元的不同模型。
随着辍学，你的神经元因此对其他特定神经元的激活变得不那么敏感，因为其他神经元可能在任何时候都被关闭。



3.1  - 具有丢失的前向传播
练习：实施具有丢失的前向传播。

您正在使用3层神经网络，并将丢弃添加到第一个和第二个隐藏层。
我们不会将失落应用于输入层或输出层。

说明：您想关闭第一层和第二层中的一些神经元。
要做到这一点，你需要执行4个步骤：

在讲座中，我们讨论了使用np.random.rand（）创建一个形状与[1]相同的变量d [1]以随机获得0和1之间的数字。
在这里，您将使用矢量化实现，因此创建一个随机矩阵D [1] = [d [1]（1）d [1]（2）... d [1]（m）]与A [1]具有相同的维数。
通过适当地对D [1]中的值进行阈值化，将D [1]的每个条目设置为概率（1-keep_prob）或概率为1（keep_prob）的1。
提示：要将矩阵X的所有条目设置为0（如果条目小于0.5）或1（如果条目大于0.5），您应该执行：X =（X <0.5）。请注意，0和1分别相当于False和True。
将A [1]设置为A [1] * D [1]。 （你正在关闭一些神经元）。你可以把D [1]看作一个掩模，以便当它与另一个矩阵相乘时，它关闭一些值。
用keep_prob除A [1]。通过这样做，您可以确保成本结果仍然具有与没有退出相同的预期价值。 （这种技术也被称为反向丢失。）
"""

def forward_propagation_with_dropout(X,parameters,keep_prob=0.5):
    """
    实现具有随机舍弃节点的前向传播。
    LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
    
    参数：
        X  - 输入数据集，维度为（2，示例数）
        parameters - 包含参数“W1”，“b1”，“W2”，“b2”，“W3”，“b3”的python字典：
            W1  - 权重矩阵，维度为（20,2）
            b1  - 偏向量，维度为（20,1）
            W2  - 权重矩阵，维度为（3,20）
            b2  - 偏向量，维度为（3,1）
            W3  - 权重矩阵，维度为（1,3）
            b3  - 偏向量，维度为（1,1）
        keep_prob  - 随机删除的概率，实数
    返回：
        A3  - 最后的激活值，维度为（1,1），正向传播的输出
        cache - 存储了一些用于计算反向传播的数值的元组
    """
    np.random.seed(1)
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    #LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1,X) + b1
    A1 = reg_utils.relu(Z1)
    
    #下面的步骤1-4对应于上述的步骤1-4。
    D1 = np.random.rand(A1.shape[0],A1.shape[1])    #步骤1：初始化矩阵D1 = np.random.rand(..., ...)
    D1 = D1 < keep_prob                             #步骤2：将D1的值转换为0或1（使​​用keep_prob作为阈值）
    A1 = A1 * D1                                    #步骤3：舍弃A1的一些节点（将它的值变为0或False）
    A1 = A1 / keep_prob                             #步骤4：缩放未舍弃的节点(不为0)的值
    """
    #不理解的同学运行一下下面代码就知道了。
    import numpy as np
    np.random.seed(1)
    A1 = np.random.randn(1,3)
    
    D1 = np.random.rand(A1.shape[0],A1.shape[1])
    keep_prob=0.5
    D1 = D1 < keep_prob
    print(D1)
    
    A1 = 0.01
    A1 = A1 * D1
    A1 = A1 / keep_prob
    print(A1)
    """
    
    Z2 = np.dot(W2,A1) + b2
    A2 = reg_utils.relu(Z2)
    
    #下面的步骤1-4对应于上述的步骤1-4。
    D2 = np.random.rand(A2.shape[0],A2.shape[1])    #步骤1：初始化矩阵D2 = np.random.rand(..., ...)
    D2 = D2 < keep_prob                             #步骤2：将D2的值转换为0或1（使​​用keep_prob作为阈值）
    A2 = A2 * D2                                    #步骤3：舍弃A1的一些节点（将它的值变为0或False）
    A2 = A2 / keep_prob                             #步骤4：缩放未舍弃的节点(不为0)的值
    
    Z3 = np.dot(W3, A2) + b3
    A3 = reg_utils.sigmoid(Z3)
    
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    
    return A3, cache

"""
3.2  - 退出传播
练习：实施具有丢失的后向传播。和以前一样，你正在训练一个3层网络。
使用存储在缓存中的掩码D [1]和D [2]将丢弃添加到第一个和第二个隐藏层。

说明：退出反向传播实际上非常简单。
你将不得不执行2个步骤：

您在前向传播期间通过将掩码D [1]应用于A1来关闭了一些神经元。
在反向传播中，通过重新应用相同的掩模D [1]到dA1，您将不得不关闭相同的神经元。
在正向传播期间，您已经通过keep_prob划分了A1。
在反向传播中，你必须再次用keep_prob除dA1（演算的解释是，如果A [1]被keep_prob调整，那么它的导数dA [1]也被相同的keep_prob调整）。
"""

def backward_propagation_with_dropout(X,Y,cache,keep_prob):
   """
    实现我们随机删除的模型的后向传播。
    参数：
        X  - 输入数据集，维度为（2，示例数）
        Y  - 标签，维度为（输出节点数量，示例数量）
        cache - 来自forward_propagation_with_dropout（）的cache输出
        keep_prob  - 随机删除的概率，实数
    
    返回：
        gradients - 一个关于每个参数、激活值和预激活变量的梯度值的字典
    """
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = (1 / m) * np.dot(dZ3,A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
    
    dA2 = dA2 * D2          #步骤1：使用正向传播期间相同的节点，舍弃那些关闭的节点（因为任何数乘以0或者False都为0或者False）
    dA2 = dA2 / keep_prob   # 步骤2：缩放未舍弃的节点(不为0)的值
    
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    
    dA1 = dA1 * D1          #步骤1：使用正向传播期间相同的节点，舍弃那些关闭的节点（因为任何数乘以0或者False都为0或者False）
    dA1 = dA1 / keep_prob   # 步骤2：缩放未舍弃的节点(不为0)的值

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients

"""
现在让我们用dropout运行模型（keep_prob = 0.86）。
这意味着在每次迭代中，您都可以24％的概率关闭第1层和第2层的每个神经元。函数model（）现在将调用：

使用forward_propagation_with_dropout而不是forward_propagation。
使用backward_propagation_with_dropout而不是backward_propagation。
"""

parameters = model(train_X, train_Y, keep_prob=0.86, learning_rate=0.3,is_plot=False)

print("使用随机删除节点，训练集:")
predictions_train = reg_utils.predict(train_X, train_Y, parameters)
print("使用随机删除节点，测试集:")
reg_utils.predictions_test = reg_utils.predict(test_X, test_Y, parameters)

"""
第0次迭代，成本值为：0.654391240515
第10000次迭代，成本值为：0.0610169865749
第20000次迭代，成本值为：0.0605824357985

使用随机删除节点，训练集:
Accuracy: 0.928909952607
使用随机删除节点，测试集:
Accuracy: 0.95

"""

plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)

"""
请注意，正规化会伤害训练集的表现！这是因为它限制了网络过度适应训练集的能力。但由于它最终提供了更好的测试准确性，因此它正在帮助您的系统。

祝贺你完成这项任务！也为法国足球革命化。 :-)
"""

'''

"""#-----------------------------------------------------------梯度校验-----------------------------------------------------------#"""


"""
欢迎来到本周的最终作业！在此作业中，您将学习实施并使用梯度检查。

您是全球范围内致力于开展移动支付的团队的一员，并被要求建立一个深度学习模型来检测欺诈行为 - 每当有人付款时，您都希望看到付款是否有欺诈行为，

例如用户帐户已被黑客接管。

但后向传播实施起来相当具有挑战性，并且有时会出现错误。

由于这是关键任务应用程序，因此贵公司的首席执行官希望确保您的反向传播实施是正确的。

你的首席执行官说：“给我一个证明你的反向传播实际上是有效的！”为了让这个保证，你将使用“梯度检查”。

我们开始做吧！

1）梯度检查如何工作？
反向传播计算梯度∂J∂θ∂J∂θ，其中θθ表示模型的参数。 JJ使用正向传播和损失函数进行计算。

因为向前传播相对容易实现，所以您确信自己得到了正确的结果，所以您几乎100％确定您正确计算了JJ的成本。

因此，您可以使用您的代码来计算JJ来验证计算的代码∂J∂θ∂J∂θ。

让我们回头看一下导数（或梯度）的定义：

如果你不熟悉“lim→→0limε→0”表示法，这只是一种说法，“当εε真的很小时”。

我们知道以下几点：

∂J∂θ∂J∂θ是你想确保你正确计算的东西。
您可以计算J（θ+ε）J（θ+ε）和J（θ-ε）J（θ-ε）（在θθ是实数的情况下），因为您确信JJ的实现方式是正确。
让我们使用方程（1）和一个εε的小数值来说服你的CEO，你的计算代码∂J∂θ∂J∂θ是正确的！


2）1-维梯度检查
考虑一维线性函数J（θ）=θxJ（θ）=θx。该模型仅包含单个实值参数θθ，并将xx作为输入。

您将实现代码来计算J（。）J（。）及其导数∂J∂θ∂J∂θ。然后，您将使用梯度检查来确保JJ的派生计算是正确的。

他上面的图表显示了关键的计算步骤：首先从xx开始，然后评估函数J（x）J（x）（“前向传播”）。然后计算导数∂J∂θ∂J∂θ（“反向传播”）。

练习：实现这个简单功能的“前向传播”和“后向传播”。即，在两个单独的函数中计算J（。）J（。）（“前向传播”）及其相对于θθ（“后向传播”）的导数。

"""

def forward_propagation(x,theta):
    """
    
    实现图1中呈现的线性前向传播（计算J）（J（theta）= theta * x）
    
    参数：
    x  - 一个实值输入
    theta  - 我们的参数，也是一个实数
    
    返回：
    J  - 函数J的值，用公式J（theta）= theta * x计算
    """
    J = np.dot(theta,x)
    
    return J

'''
#测试forward_propagation
print("-----------------测试forward_propagation-----------------")
x, theta = 2, 4
J = forward_propagation(x, theta)
print ("J = " + str(J))
"""
-----------------测试forward_propagation-----------------
J = 8
"""
'''

"""

练习：现在，执行图1的后向传播步骤（微分计算）。
也就是说，计算相对于θθ的J（θ）=θxJ（θ）=θx的导数。
为了让你免于微积分，你应该得到dtheta =∂J∂θ= xdtheta =∂J∂θ= x。
"""

def backward_propagation(x,theta):
    """
    计算J相对于θ的导数（见图1）。
    
    参数：
        x  - 一个实值输入
        theta  - 我们的参数，也是一个实数
    
    返回：
        dtheta  - 相对于θ的成本梯度
    """
    dtheta = x
    
    return dtheta

#测试backward_propagation
print("-----------------测试backward_propagation-----------------")
x, theta = 2, 4
dtheta = backward_propagation(x, theta)
print ("dtheta = " + str(dtheta))

"""
-----------------测试backward_propagation-----------------
dtheta = 2
"""

"""
练习：为了显示backward_propagation（）函数正确计算梯度∂J∂θ∂J∂θ，让我们执行梯度检查。

说明：

首先使用上面的公式（1）和εε的一个小值计算“gradapprox”。以下是要遵循的步骤：
θ+ =θ+εθ+ =​​θ+ε
θ-=θ-εθ-=θ-ε
J + = j的（θ+）J + = j的（θ+）
J- = j的（θ-）J- = j的（θ-）
gradapprox = J + -J-2εgradapprox= J + -J-2ε
然后使用反向传播计算梯度，并将结果存储在变量“grad”中
最后，使用以下公式计算“gradapprox”和“grad”之间的相对差异：
差= ||grad-gradapprox||2||grad||2 + ||gradapprox||2（2）
（2）差值= ||grad-gradapprox||2||grad||2 + ||gradapprox||2
 
你将需要3个步骤来计算这个公式：
1。使用np.linalg.norm（...）计算分子
2' 。计算分母。您需要两次调用np.linalg.norm（...）。
3' 。将它们分开。
如果这种差异很小（比如说小于10-710-7），那么您可以相当确信您已经正确计算了您的梯度。否则，梯度计算可能会出现错误。

"""

def gradient_check(x,theta,epsilon=1e-7):
    """
    
    实现图1中的反向传播。
    
    参数：
        x  - 一个实值输入
        theta  - 我们的参数，也是一个实数
        epsilon  - 使用公式（1）计算输入的微小偏移以计算近似梯度
    
    返回：
        近似梯度和后向传播梯度之间的差异（2）
    """
    
    #使用公式（1）的左侧计算gradapprox。 epsilon足够小，您无需担心限制。
    thetaplus = theta + epsilon                               # Step 1
    thetaminus = theta - epsilon                              # Step 2
    J_plus = forward_propagation(x, thetaplus)                # Step 3
    J_minus = forward_propagation(x, thetaminus)              # Step 4
    gradapprox = (J_plus - J_minus) / (2 * epsilon)           # Step 5
    
    
    #检查gradapprox是否足够接近backward_propagation（）的输出
    grad = backward_propagation(x, theta)
    
    numerator = np.linalg.norm(grad - gradapprox)                      # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)    # Step 2'
    difference = numerator / denominator                               # Step 3'
    
    if difference < 1e-7:
        print("梯度检查：梯度正常!")
    else:
        print("梯度检查：梯度超出阈值!")
    
    return difference

'''

#测试gradient_check
print("-----------------测试gradient_check-----------------")
x, theta = 2, 4
difference = gradient_check(x, theta)
print("difference = " + str(difference))

"""
-----------------测试gradient_check-----------------
梯度检查：梯度正常!
difference = 2.91933588329e-10
"""
'''

"""
恭喜，差距小于10-710-7的门槛。因此，您可以高度肯定自己已正确计算了backward_propagation（）中的渐变。

现在，在更一般的情况下，您的成本函数JJ具有多于一个一维输入。
在训练神经网络时，θθ实际上由多个矩阵W [1] W [1]和偏差b [1] b [1]组成！知道如何对高维输入进行梯度检查很重要。我们开始做吧！
"""

def forward_propagation_n(X,Y,parameters):
    """
    实现图3中的前向传播（并计算成本）。
    
    参数：
        X - 训练集为m个例子
        Y -  m个示例的标签
        parameters - 包含参数“W1”，“b1”，“W2”，“b2”，“W3”，“b3”的python字典：
            W1  - 形状的权重矩阵（5,4）
            b1  - 形状的偏向量（5,1）
            W2  - 形状的权重矩阵（3,5）
            b2  - 形状（3,1）的偏向量
            W3  - 形状的权重矩阵（1,3）
            b3  - 形状（1,1）的偏向量
    
    返回：
        cost - 成本函数（logistic）
    """
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1,X) + b1
    A1 = gc_utils.relu(Z1)
    
    Z2 = np.dot(W2,A1) + b2
    A2 = gc_utils.relu(Z2)
    
    Z3 = np.dot(W3,A2) + b3
    A3 = gc_utils.sigmoid(Z3)
    
    #计算成本
    logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = (1 / m) * np.sum(logprobs)
    
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return cost, cache

def backward_propagation_n(X,Y,cache):
    """
    实现图2中所示的反向传播。
    
    参数：
        X - 形输入数据点（输入大小，1）
        Y - 真正的“标签”
        cache - 来自forward_propagation_n（）的缓存输出
    
    返回：
        gradients - 一个字典，其中包含与每个参数，激活和激活前变量相关的成本梯度。
    """
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = (1. / m) * np.dot(dZ3,A2.T)
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    #dW2 = 1. / m * np.dot(dZ2, A1.T) * 2  # Should not multiply by 2
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    #db1 = 4. / m * np.sum(dZ1, axis=1, keepdims=True) # Should not multiply by 4
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
 
    return gradients

"""
您在欺诈检测测试集中获得了一些结果，但您并非100％确定您的模型。

没有人是完美的！让我们执行渐变检查以验证您的渐变是否正确。

梯度检查如何工作？

如1）和2）中所述，您想比较“gradapprox”与反向传播计算的梯度。该公式仍然是：

然而，θθ不再是标量。这是一个名为“参数”的字典。

我们为你实现了一个函数“dictionary_to_vector（）”。

它将“参数”字典转换为称为“值”的向量，通过将所有参数（W1，b1，W2，b2，W3，b3）整形为向量并将它们连接起来而获得。

反函数是“vector_to_dictionary”，它返回“参数”字典。


我们还使用gradients_to_vector（）将“gradients”字典转换为矢量“grad”。你不必担心这一点。

练习：实施gradient_check_n（）。

说明：这里是伪码，可以帮助你实现梯度检查。

对于num_parameters中的每个i：

要计算J_plus [i]：
将θ+θ+设置为np.copy（parameters_values）
将θ+iθi+设置为θ+ i +εθi++ε
使用forward_propagation_n（x，y，vector_to_dictionary（θ+θ+））计算J + iJi +。
计算J_minus [i]：用θ-θ-做同样的事情，
计算gradapprox [i] = J + i-J-i2εgradapprox[i] = Ji + -Ji-2ε
因此，您会得到一个矢量gradapprox，其中gradapprox [i]是相对于parameter_values [i]的梯度近似值。
您现在可以将此Gradapprox矢量与反向传播的梯度矢量进行比较。就像一维情况一样（步骤1'，2'，3'），计算：


"""

def gradient_check_n(parameters,gradients,X,Y,epsilon=1e-7):
    """
    检查backward_propagation_n是否正确计算forward_propagation_n输出的成本梯度
    
    参数：
        parameters - 包含参数“W1”，“b1”，“W2”，“b2”，“W3”，“b3”的python字典：
        grad_output_propagation_n的输出包含与参数相关的成本梯度。
        x  - 输入数据点，形状（输入大小，1）
        y  - 真正的“标签”
        epsilon  - 使用公式（1）计算输入的微小偏移以计算近似梯度
    
    返回：
        difference - 近似梯度和后向传播梯度之间的差异（2）
    """
    #初始化参数
    parameters_values , keys = gc_utils.dictionary_to_vector(parameters) #keys用不到
    grad = gc_utils.gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters,1))
    J_minus = np.zeros((num_parameters,1))
    gradapprox = np.zeros((num_parameters,1))
    
    #计算gradapprox
    for i in range(num_parameters):
        #计算J_plus [i]。输入：“parameters_values，epsilon”。输出=“J_plus [i]”
        thetaplus = np.copy(parameters_values)                                                  # Step 1
        thetaplus[i][0] = thetaplus[i][0] + epsilon                                             # Step 2
        J_plus[i], cache = forward_propagation_n(X,Y,gc_utils.vector_to_dictionary(thetaplus))  # Step 3 ，cache用不到
        
        #计算J_minus [i]。输入：“parameters_values，epsilon”。输出=“J_minus [i]”。
        thetaminus = np.copy(parameters_values)                                                 # Step 1
        thetaminus[i][0] = thetaminus[i][0] - epsilon                                           # Step 2        
        J_minus[i], cache = forward_propagation_n(X,Y,gc_utils.vector_to_dictionary(thetaminus))# Step 3 ，cache用不到
        
        #计算gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
        
    #通过计算差异比较gradapprox和后向传播梯度。
    numerator = np.linalg.norm(grad - gradapprox)                                     # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)                   # Step 2'
    difference = numerator / denominator                                              # Step 3'
    
    if difference < 1e-7:
        print("梯度检查：梯度正常!")
    else:
        print("梯度检查：梯度超出阈值!")
    
    return difference

"""

看来我们给你的backward_propagation_n代码有错误！很好，你已经实施了梯度检查。

返回到backward_propagation并尝试查找/更正错误（提示：检查dW2和db1）。

当你认为你已经修复它时重新运行渐变检查。

记住，如果修改代码，则需要重新执行定义backward_propagation_n（）的单元格。

你可以通过渐变检查来声明你的派生计算是正确的吗？即使这部分任务没有分级，我们强烈建议您尝试找到bug并重新运行梯度检查，直到您确信backprop现在已正确实施。

注意

渐变检查很慢！以近似的梯度为计算成本很高，其中∂J∂θ≈J（θ+ε）-J（θ-ε）2ε∂J∂θ≈J（θ+ε）-J（θ-ε）2ε。出于这个原因，我们不会在训练期间的每次迭代中运行梯度检查。只需几次检查梯度是否正确。
渐变检查，至少在我们提出的时候，不适用于退出。您通常会运行渐变检查算法而不丢失，以确保您的backprop是正确的，然后添加丢失。
恭喜，您可以确信您的欺诈检测深度学习模式正常运行！你甚至可以用它来说服你的首席执行官。 :)
"""









































































