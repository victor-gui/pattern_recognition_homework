# -*- coding: utf-8 -*-

import numpy as np
import h5py
import matplotlib.pyplot as plt
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# index = 200
# plt.imshow(train_set_x_orig[index])
# print("train_set_y=" + str(train_set_y))
# plt.show()

#打印出当前的训练标签值
#使用np.squeeze的目的是压缩维度，【未压缩】train_set_y[:,index]的值为[1] , 【压缩后】np.squeeze(train_set_y[:,index])的值为1
#print("【使用np.squeeze：" + str(np.squeeze(train_set_y[:,index])) + "，不使用np.squeeze： " + str(train_set_y[:,index]) + "】")
#只有压缩后的值才能进行解码操作
#print("y=" + str(train_set_y[:,index]) + ", it's a " + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") + "' picture")

m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px1 = train_set_x_orig.shape[1]
num_px2 = train_set_x_orig.shape[2]

print("训练集的数量：m_train= "+str(m_train))
print("测试集的数量：m_test= "+str(m_test))
print("每张图片的宽/高：num_px= "+str(num_px1)+"/"+str(num_px2))
print("每张图片的大小：（"+str(num_px1)+","+str(num_px2)+",3)")
print("训练集_图片的维数："+str(train_set_x_orig.shape))
print("训练集_标签的维数："+str(train_set_y.shape))
print("测试集_图片的维数："+str(test_set_x_orig.shape))
print("测试集_标签的维数："+str(test_set_y.shape))

#将训练集和测试集的维度降低并转置
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print ("训练集降维最后的维度： " + str(train_set_x_flatten.shape))
print ("训练集_标签的维数 : " + str(train_set_y.shape))
print ("测试集降维之后的维度: " + str(test_set_x_flatten.shape))
print ("测试集_标签的维数 : " + str(test_set_y.shape))

#标准化数据集
train_set_x=train_set_x_flatten/255
test_set_x=test_set_x_flatten/255

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

'''
#测试sigmoid()
print("====================测试sigmoid====================")
print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(9.2) = " + str(sigmoid(9.2)))
'''

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return (w,b)

#正向传播与反向传播
def propagate(w, b, X, Y):
    m = X.shape[1]
#正向
    A = sigmoid(np.dot(w.T,X)+b)
    cost = (-1/m)*np.sum(Y*np.log(A)+(1-Y)*(np.log(1-A)))
#反向
    dw = (1/m)*(np.dot(X,(A-Y).T))
    db = (1/m)*(np.sum(A-Y))

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

#创建一个字典保存dw,db
    grads={
            'dw': dw,
            'db': db
          }

    return (grads,cost)

'''
#测试一下propagate
print("====================测试propagate====================")
#初始化一些参数
w, b, X, Y = np.array([[1],[2],[3]]), 2, np.array([[1,2],[2,3],[3,2]]), np.array([[2,1]])
grads, cost = propagate(w,b,X,Y)
print("dw= "+str(grads['dw']))
print("db= "+str(grads['db']))
print("cost= "+str(cost))
'''

def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):

    costs = []

    for i in range(num_iterations):
        grads,cost = propagate(w,b,X,Y)
        dw = grads['dw']
        db = grads['db']

        w = w-learning_rate*dw
        b = b-learning_rate*db

        if i%100 == 0:
            costs.append(cost)
        if (print_cost) and (i%100 == 0):
            print("迭代次数：%d ，误差值：%f" % (i,cost))

    params = {
              'w': w,
              'b': b
             }
    grads = {
              'dw': dw,
              'db': db
            }

    return (params, grads, costs)

'''
#测试optimize
print("====================测试optimize====================")
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
params , grads , costs = optimize(w , b , X , Y , num_iterations=1000 , learning_rate = 0.009 , print_cost =True)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
'''

def predict(w,b,X):

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)

    A = sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        if A[0,i]>0.5:
            Y_prediction[0,i]=1
        else:
            Y_prediction[0,i]=0

    assert(Y_prediction.shape == (1,m))

    return Y_prediction

'''
#测试predict
print("====================测试predict====================")
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
print("predictions = " + str(predict(w, b, X)))
'''

#主程序
def model(X_train,Y_train,X_test,Y_test,num_iterations=2000,learning_rate=0.5,print_cost=False):

    w, b = initialize_with_zeros(X_train.shape[0])

    parameters,grads,costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    w, b = parameters['w'],parameters['b']

    Y_prediction_train = predict(w,b,X_train)
    Y_prediction_test = predict(w,b,X_test)

    print("训练集的准确度为：",format(100-np.mean(np.abs(Y_prediction_train-Y_train)*100)),"%")
    print("测试集的准确度为：",format(100-np.mean(np.abs(Y_prediction_test-Y_test) * 100)), "%")

    d = {
            'costs': costs,
            'Y_prediction_train': Y_prediction_train,
            'Y_prediction_test': Y_prediction_test,
            'w': w,
            'b': b,
            'num_iterations': num_iterations,
            'learning_rate': learning_rate}
    return d


print("============测试model=============")
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

#绘图
print("=============绘图啦==============")

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.xlabel("iterations(per hundred)")
plt.ylabel("costs")
plt.title("learning_rate= "+str(d['learning_rate']))
plt.show()