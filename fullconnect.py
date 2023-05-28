# 手写数字识别问题
# 全连接神经网络
import keras.callbacks
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation


(X_train,Y_train),(X_test,Y_test)=mnist.load_data()
X_train=X_train.reshape(60000,784).astype('float32')/255.0
X_test=X_test.reshape(10000,784).astype('float32')/255.0
Y_train=np_utils.to_categorical(Y_train)
Y_test=np_utils.to_categorical(Y_test)

model=Sequential()
model.add(Dense(units=64,input_dim=28*28,activation='relu'))
model.add(Dense(units=10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics='accuracy')

tb_hist=keras.callbacks.TensorBoard(log_dir='./graph',histogram_freq=0,write_graph=True,write_images=True)
model.fit(X_train,Y_train,epochs=5,batch_size=32,callbacks=[tb_hist])


# 评估:得到损失以及精度
loss_and_metrics=model.evaluate(X_test,Y_test, batch_size=32)
print ('loss_and_metrics:'+str(loss_and_metrics))
model.save('./test/fullconnect.model')
