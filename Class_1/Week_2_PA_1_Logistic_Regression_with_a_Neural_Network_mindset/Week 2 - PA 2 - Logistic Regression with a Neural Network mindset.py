# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 16:32:27 2018

@author: Oscar
"""

"""

"""
import numpy as np
import matplotlib.pyplot as plt
from lr_utils import load_dataset

"""
train_set_x_orig和test_set_x_orig是表示图像的数组

train_set_y和test_set_y是【0 | 1】标签

classes = [b'non-cat' b'cat']

我们在图像数据集的最后添加了“_orig”（训练和测试），

因为我们要对它们进行预处理。

预处理后，我们将以train_set_x和test_set_x结束

（标签train_set_y和test_set_y不需要任何预处理）。

train_set_x_orig和test_set_x_orig的每一行都是表示图像的数组。

您可以通过运行以下代码来查看示例。随意更改索引值并重新运行以查看其他图像。
"""

train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = load_dataset()
#print(str(classes),type(classes))


index = 25
#plt.imshow(train_set_x_orig[index])
#print("train_set_y=" + str(train_set_y))
"""
train_set_y=[[0 0 1 0 0 0 0 1 0 0 0 1 0 1 1 0 0 0 0 1 0 0 0 0 1 1 0 1 0 1 0 0 0 0 0 0 0
              0 1 0 0 1 1 0 0 0 0 1 0 0 1 0 0 0 1 0 1 1 0 1 1 1 0 0 0 0 0 0 1 0 0 1 0 0
              0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 0 0 0 1 1 1 0 0 1 0 0 0 0 1 0 1 0 1 1 1 1 1
              1 0 0 0 0 0 1 0 0 0 1 0 0 1 0 1 0 1 1 0 0 0 1 1 1 1 1 0 0 0 0 1 0 1 1 1 0
              1 1 0 0 0 1 0 0 1 0 0 0 0 0 1 0 1 0 1 0 0 1 1 1 0 0 1 1 0 1 0 1 0 0 0 0 0
              1 0 0 1 0 0 0 1 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0]]

classes=[b'non-cat' b'cat']

"""

#打印出当前的训练标签值
#使用np.squeeze的目的是压缩维度，【未压缩】train_set_y[:,index]的值为[1] , 【压缩后】np.squeeze(train_set_y[:,index])的值为1
#print("【使用np.squeeze：" + str(np.squeeze(train_set_y[:,index])) + "，不使用np.squeeze： " + str(train_set_y[:,index]) + "】")
#只有压缩后的值才能进行解码操作
#print("y=" + str(train_set_y[:,index]) + ", it's a " + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") + "' picture")

"""
深度学习中的许多软件错误来自不适合的矩阵/矢量尺寸。

如果你能保持你的矩阵/矢量维度直，你会消除很多错误很长的路要走。

 -  m_train（训练示例的数量）
 -  m_test（测试例数）
 -  num_px（=高度=训练图像的宽度）
 
请记住，train_set_x_orig是一个形状为numpy的数组（m_​​train，num_px，num_px，3）。

可以通过编写train_set_x_orig.shape [0]来访问m_train。
"""

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


"""
为了方便起见，现在应该在shape（num_px * num_px * 3，1）的numpy数组中重塑shape（num_px，num_px，3）的图像。

在此之后，我们的训练（和测试）数据集是一个numpy数组，每列代表一个平坦的图像。应该有m_train（分别是m_test）列。

重塑训练和测试数据集，以便将大小（num_px，num_px，3）的图像展平为单个形状矢量（num_px * num_px * 3，1）。

当你想将形状（a，b，c，d）的矩阵X平铺成形状（b * c * d，a）的矩阵X_flatten时，一个技巧就是使用：

X_flatten = X.reshape（X.shape [0]，-1）.T＃X.T是X的转置
"""

#将训练集的维度降低，降低到一维,并转置
print("train_set_x_orig.shape = " , train_set_x_orig.shape)
#这一段意思是指把数组变为209行的矩阵，但是我懒得算列有多少，于是我就用-1告诉程序你帮我算，
#最后程序算出来时12288列，我在最后用一个T表示转置，这就变成了12288行，209列。
train_set_x_flatten  = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
#这一行代码意思是和上一段差不多
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print ("训练集降维最后的维度： " + str(train_set_x_flatten.shape))
print ("训练集_标签的维数 : " + str(train_set_y.shape))
print ("测试集降维之后的维度: " + str(test_set_x_flatten.shape))
print ("测试集_标签的维数 : " + str(test_set_y.shape))
print ("正确性检查 : " + str(train_set_x_flatten[0:5,0]))

"""
为了表示彩色图像，必须为每个像素指定红色，绿色和蓝色通道（RGB），

因此像素值实际上是从0到255范围内的三个数字的向量。

机器学习中一个常见的预处理步骤是对数据集进行居中和标准化，

这意味着可以减去每个示例中整个numpy数组的平均值，

然后将每个示例除以整个numpy数组的标准偏差。

但对于图片数据集，它更简单，更方便，几乎可以将数据集的每一行除以255（像素通道的最大值）。

现在标准化我们的数据集
"""

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


"""
建立神经网络的主要步骤是：

1.定义模型结构（例如输入特征的数量）

2.初始化模型的参数

3.循环：

    计算当前损失（正向传播）
    
    计算当前梯度（反向传播）
    
    更新参数（梯度下降）
    
您经常分别构建1-3并将它们集成到一个我们称为model（）的函数中。

现在构建sigmod()，需要计算 sigmoid（w ^ T x + b）来做出预测。

"""

def sigmoid(z):
    """
        参数：
        x  - 任何大小的标量或numpy数组。
    
        返回：
        s  -  sigmoid（z）
    """
    s = 1 / (1 + np.exp(-z))
    return s

#测试sigmoid()
print("====================测试sigmoid====================")
print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(9.2) = " + str(sigmoid(9.2)))


"""
初始化参数
"""
def initialize_with_zeros(dim):
    """
        此函数为w创建一个维度为（dim，1）的0向量，并将b初始化为0。
        
        论据：
            dim  - 我们想要的w矢量的大小（或者这种情况下的参数数量）
        
        返回：
            w  - 形状的初始化向量（dim，1）
            b  - 初始化的标量（对应于偏差）
    """
    w = np.zeros(shape = (dim,1))
    b = 0
    #使用断言来确保我要的数据是正确的
    assert(w.shape == (dim, 1)) #w的维度是(dim,1)
    assert(isinstance(b, float) or isinstance(b, int)) #b的类型是float或者int
    
    return (w , b)


"""
向前和向后传播

现在我的参数已初始化，可以执行“前进”和“后退”传播步骤来学习参数。

实现一个计算成本函数及其渐变的函数propagate（）。

向前传播：

"""

def propagate(w, b, X, Y):
    """
    实现上述传播的成本函数及其梯度

    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 矩阵类型为（num_px * num_px * 3，训练数量）
        Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据数量)

    返回：
        成本 - 逻辑回归的负对数似然成本
        dw  - 相对于w的损失梯度，因此与w相同的形状
        db  - 相对于b的损失梯度，因此与b的形状相同
    """
    
    m = X.shape[1]
    
    #正向传播
    A = sigmoid(np.dot(w.T,X) + b) #计算激活值
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A))) #计算成本
    
    #反向传播
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    
    #使用断言确保我的数据是正确的
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {
                "dw": dw,
                "db": db
             }
    return (grads , cost)

''' 
#测试一下propagate
print("====================测试propagate====================")
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))
'''
    
"""
优化
您已初始化您的参数。
您还可以计算成本函数及其梯度。
现在，您要使用渐变下降更新参数。
练习：写下优化功能。
目标是通过最小化成本函数 J 来学习w 和 b 。
对于参数  theta ，
更新规则是 theta = theta  - alpha * d * theta ，
其中  alpha 是学习率。
"""

def optimize(w , b , X , Y , num_iterations , learning_rate , print_cost = False):
    """
    此函数通过运行梯度下降算法来优化w和b
    
    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 形状数据（num_px * num_px * 3，训练数据数量）
        Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据数量)
        num_iterations  - 优化循环的迭代次数
        learning_rate  - 梯度下降更新规则的学习率
        print_cost  - 每100步打印一次损失
    
    返回：
        params  - 包含权重w和偏差b的字典
        grads  - 包含权重和偏差相对于成本函数的梯度的字典
        成本 - 优化期间计算的所有成本列表，这将用于绘制学习曲线。
    
    提示：
    你基本上需要写下两个步骤并遍历它们：
        1）计算当前参数的成本和梯度。使用propagate（）。
        2）使用w和b的梯度下降法则更新参数。
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        grads, cost = propagate(w, b, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        #记录成本
        if i % 100 == 0:
            costs.append(cost)
        #打印成本数据
        if (print_cost) and (i % 100 == 0):
            print("迭代的次数: %i ， 误差值： %f" % (i,cost))
        
    params  = {
                "w" : w,
                "b" : b }
    grads = {
            "dw": dw,
            "db": db } 
    return (params , grads , costs)

'''
#测试optimize
print("====================测试optimize====================")
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
params , grads , costs = optimize(w , b , X , Y , num_iterations=100 , learning_rate = 0.009 , print_cost = False)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
'''
"""
前一个函数将输出已学习的w和b。

我们可以使用w和b来预测数据集X的标签。

实现predict（）函数。计算预测有两个步骤：

计算 hat {Y} = A = sigma（w ^ T  * X + b）

将a的条目转换为0（如果激活值<= 0.5）或1（如果激活值> 0.5），

则将预测值存储在向量Y_prediction中。

如果你愿意的话，你可以在for循环中使用if / else语句（虽然也有一种方法可以对此进行矢量化）。

"""

def predict(w , b , X ):
    """
    使用学习逻辑回归参数logistic （w，b）预测标签是0还是1，
    
    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 大小数据（num_px * num_px * 3，训练数据数量）
    
    返回：
        Y_prediction  - 包含X中所有例子的所有预测（0/1）的一个numpy数组（向量）
    
    """
    
    m  = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    
    #计算向量“A”预测猫在图片中出现的概率
    A = sigmoid(np.dot(w.T , X) + b)
    for i in range(A.shape[1]):
        #将概率a [0，i]转换为实际预测p [0，i]
        Y_prediction[0,i] = 1 if A[0,i] > 0.5 else 0
    assert(Y_prediction.shape == (1,m))
    
    return Y_prediction
'''

#测试predict
print("====================测试predict====================")       
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]]) 
print("predictions = " + str(predict(w, b, X)))
'''

"""
将所有功能合并到模型中

现在将看到整个模型是如何构建的，将所有构建模块（前面部分中实现的功能）以正确的顺序放在一起。

实现模型功能。使用以下表示法：

 -  Y_prediction您对测试集的预测
 -  Y_prediction_train在火车上的预测
 -  w，成本，优化（）的输出的梯度
"""

def model(X_train , Y_train , X_test , Y_test , num_iterations = 2000 , learning_rate = 0.5 , print_cost = False):
    """
    通过调用之前实现的函数来构建逻辑回归模型
    
    参数：
        X_train  - numpy的数组,维度为（num_px * num_px * 3，m_train）表示的训练集
        Y_train  - numpy的数组,维度为（1，m_train）（矢量）表示的训练标签
        X_test   - numpy的数组,维度为（num_px * num_px * 3，m_test）表示的测试集
        Y_test   - numpy的数组,维度为（1，m_test）的（向量）表示的测试标签
        num_iterations  - 表示用于优化参数的迭代次数的超参数
        learning_rate  - 表示optimize（）更新规则中使用的学习速率的超参数
        print_cost  - 设置为true以每100次迭代打印成本
    
    返回：
        d  - 包含有关模型信息的字典。
    """
    w , b = initialize_with_zeros(X_train.shape[0])
    
    parameters , grads , costs = optimize(w , b , X_train , Y_train,num_iterations , learning_rate , print_cost)
    
    #从字典“参数”中检索参数w和b
    w , b = parameters["w"] , parameters["b"]
    #预测测试/训练集的例子
    Y_prediction_test = predict(w , b, X_test)
    Y_prediction_train = predict(w , b, X_train)
    
    #打印训练后的准确性
    print("训练集准确性："  , format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100) ,"%")
    print("测试集准确性："  , format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100) ,"%")
    
    d = {
            "costs" : costs,
            "Y_prediction_test" : Y_prediction_test,
            "Y_prediciton_train" : Y_prediction_train,
            "w" : w,
            "b" : b,
            "learning_rate" : learning_rate,
            "num_iterations" : num_iterations }
    return d

print("====================测试model====================")     
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)


'''
#绘制图
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
'''

"""
您可以看到成本下降。它显示参数正在被学习。

但是，您发现您可以在训练集上更多地训练模型。

尝试增加上面单元格中的迭代次数并重新运行单元格。

您可能会看到训练集的准确度上升，但测试集准确度下降。这被称为过度拟合。


恭喜您构建您的第一个图像分类模型。

让我们进一步分析一下，并研究学习率alpha的可能选择。

学习率的选择

提醒：为了让渐变下降起作用，您必须明智地选择学习速率。

学习率alpha 决定了我们更新参数的速度。

如果学习率过高，我们可能会“超过”最优值。

同样，如果它太小，我们将需要太多迭代才能收敛到最佳值。

这就是为什么使用良好调整的学习率至关重要。

让我们比较一下我们模型的学习曲线和几种学习速率的选择。

也可以尝试使用不同于我们初始化的learning_rates变量包含的三个值，并查看会发生什么。
"""

learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

'''
"""
不同的学习率会产生不同的成本，从而导致预测结果不同。

如果学习率过高（0.01），成本可能会上下波动。它甚至可能会有分歧（尽管在这个例子中，使用0.01仍然最终会以成本为代价）。

较低的成本并不意味着更好的模式。你必须检查是否有可能过度配合。当训练的准确性远高于测试的准确性时，会发生这种情况。

在深度学习中，我们通常建议您：

选择更好地最小化成本函数的学习率。

如果您的模型过度配合，请使用其他技术来减少过度配合。 （我们将在稍后的视频中讨论这个问题。）

7  - 用自己的图片进行测试（可选/未评分练习）

祝贺您完成这项任务。您可以使用自己的图像并查看模型的输出。要做到这一点：

1.单击本笔记本上部栏中的“文件”，然后单击“打开”以进入Coursera Hub。
2.将图像添加到Jupyter Notebook的目录中的“images”文件夹中
3.在下面的代码中更改图像的名称
4.运行代码并检查算法是否正确（1 = cat，0 =非cat）！
"""








'''























