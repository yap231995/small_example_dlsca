


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical


def mlp(length, classes=256, lr=0.1):
    img_input = Input(shape=(length, ))
    x = Dense(2, activation='sigmoid')(img_input)
    # x = Dense(classes, activation='softmax')(x)
    x = Dense(1, activation = 'sigmoid')(x)
    model = Model(img_input, x)
    # optimizer = Adam(lr=lr)
    optimizer = RMSprop(lr=lr)
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

seed = 100
tf.random.set_seed(seed)
classes = 2
traces = np.array([[0,0],[0,1],[1,0],[1,1]], dtype = np.int8)
label = np.array([0,1,1,0], dtype = np.int8)
# label = np.array([[0],[1],[1],[0]], dtype = np.int8)
one_hot_label = label
# one_hot_label = to_categorical(label, num_classes=classes)
# one_hot_label = np.array([[0.8,0.2],[0.2,0.8],[0.2,0.8],[0.8,0.2]])
# print(one_hot_label)
input_length = traces.shape[1]

model = mlp(input_length, classes)
model.fit(x = traces, y = one_hot_label, batch_size= 4, verbose=2, epochs=2000)
# for layer in model.layers:
#     print("Layer: ", layer)
#     weights = layer.weights
#     # print("weight: ", weights)
#     # print("bias: ", weights[2])


y0 = model.predict(np.array([[0,0]]))
y1 = model.predict(np.array([[0,1]]))
y2 = model.predict(np.array([[1,0]]))
y3 = model.predict(np.array([[1,1]]))

print("x = [0,0], y = [1,0]", y0)
print("x = [0,1], y = [0,1]", y1)
print("x = [1,0], y = [1,0]", y2)
print("x = [1,1], y = [0,1]", y3)




