#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# keras imports for the dataset and building our neural network
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils


# In[2]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[3]:


# let's print the shape before we reshape and normalize
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

# building the input vector from the 28x28 pixels
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# print the final input shape ready for training
print("Train matrix shape", X_train.shape)
print("Test matrix shape", X_test.shape)


# In[4]:


print(np.unique(y_train, return_counts=True))


# In[5]:


# one-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)


# In[6]:


# building a linear stack of layers with the sequential model
model = Sequential()
model.add(Dense(128, input_shape=(784,)))
model.add(Activation('relu'))                            



model.add(Dense(10))
model.add(Activation('softmax'))


# In[7]:


# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


# In[8]:


# training the model and saving metrics in history
history = model.fit(X_train, Y_train,
          batch_size=1024, epochs=2,
          verbose=2,
          validation_data=(X_test, Y_test))


# In[ ]:


# saving the model
model.save("keras_mnist4.h5")


# In[ ]:


mnist_model = load_model("keras_mnist4.h5")
loss_and_metrics = mnist_model.evaluate(X_test, Y_test, verbose=2)

print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])
accuracy=(loss_and_metrics[1]*100)
with open('/train/acc.txt', 'w') as f:
    f.write("%d" % accuracy)

# In[ ]:





# In[ ]:




