# coding: utf-8

import sys
import theano
import theano.tensor as T
import numpy as np
import pandas as pd
from materials import iris_dataset


# data
iris_data = iris_dataset()
iris_data = iris_data.reindex(np.random.permutation(iris_data.index))
iris_x = iris_data[iris_data.columns[:4]].as_matrix()
iris_y = pd.get_dummies(iris_data[iris_data.columns[4]]).values


input_dim = iris_x.shape[1]
hidden_dim = 9
output_dim = iris_y.shape[1]


# models
X = T.fmatrix('x')
Y = T.fmatrix('y')

W_i = theano.shared(np.random.randn(input_dim, hidden_dim), name='W')
b_i = theano.shared(np.zeros((hidden_dim,)), name='b')
W_h = theano.shared(np.random.randn(hidden_dim, output_dim), name='W')
b_h = theano.shared(np.zeros((output_dim,)), name='b')

o_h = T.nnet.sigmoid(T.dot(X, W_i) + b_i)
p_y_given_x = T.nnet.sigmoid(T.dot(o_h, W_h) + b_h)


# 训练设置
params = [W_i, b_i, W_h, b_h]
predict_func = theano.function(
    inputs=[X],
    outputs=p_y_given_x,
    allow_input_downcast=True
)
cost = T.mean((p_y_given_x - Y) ** 2)
grad = [T.grad(cost, param) for param in params]
lr = 0.3
updates = [
    (param, param - lr * gparam)
    for param, gparam in zip(params, grad)
]
train_func = theano.function(
    inputs=[X, Y],
    outputs=cost,
    updates=updates,
    allow_input_downcast=True
)


train_size = len(iris_x)
batch_size = 32
for i in range(100):
    for start in range(0, train_size, batch_size):
        batch_x = iris_x[start:start+batch_size]
        batch_y = iris_y[start:start+batch_size]
        cost_val = train_func(batch_x, batch_y)

    print '[%d]Cost:' % i, cost_val
