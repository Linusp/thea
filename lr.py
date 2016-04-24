# coding: utf-8

import theano
import theano.tensor as T
import numpy as np
import pandas as pd
from double_moon import double_moon

# origin data
DM_X, DM_Y = double_moon(20, 50)

# model
X = T.fmatrix()
Y = T.fvector()
W = theano.shared(np.random.randn(2))
p_y_given_x = T.nnet.sigmoid(T.dot(X, W))
y_pred = p_y_given_x > 0.5

cost = T.mean(T.nnet.binary_crossentropy(p_y_given_x, Y))
grad = T.grad(cost=cost, wrt=W)
update = [[W, W - grad * 0.03]]

train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

train_size = int(len(DM_X) * 0.9)
indices = np.arange(len(DM_X))
np.random.shuffle(indices)

x = DM_X[indices]
y = DM_Y[indices]

train_x = x[:train_size]
train_y = y[:train_size]
test_x = x[train_size:]
test_y = y[train_size:]

batch_size = 128
for i in range(100):
    for start in range(0, train_size, batch_size):
        cost = train(train_x[start:start+batch_size], train_y[start:start+batch_size])
        print '[%d]Cost:' % i, cost
