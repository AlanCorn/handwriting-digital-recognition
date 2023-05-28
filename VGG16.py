import cv2
import keras.callbacks
import numpy as np
from keras import Sequential
from keras.applications import VGG16
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D

from keras.optimizers import Adam
opt = Adam(lr=0.001)

#加载mnist数据集，为了缩短训练时间取数据集前10000个
(x_train,y_train),(x_test,y_test)=mnist.load_data()
# x_train,y_train=x_train[:10000],y_train[:10000]
# x_test,y_test=x_test[:10000],y_test[:10000]

#修改数据集的尺寸、将灰度图像转换为rgb图像
x_train=[cv2.cvtColor(cv2.resize(i,(48,48)),cv2.COLOR_GRAY2BGR)for i in x_train]
x_train=np.concatenate([arr[np.newaxis]for arr in x_train]).astype('float32')
x_train=x_train/255
x_test=[cv2.cvtColor(cv2.resize(i,(48,48)),cv2.COLOR_GRAY2BGR)for i in x_test]
x_test=np.concatenate([arr[np.newaxis]for arr in x_test]).astype('float32')
x_test=x_test/255

#编码
y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)

model=VGG16(include_top=False,weights="imagenet",input_shape=(48,48,3))

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')

#训练模型
tb_hist=keras.callbacks.TensorBoard(log_dir='./graph',histogram_freq=0,write_graph=True,write_images=True)
model.fit(x_train,y_train,epochs=3,batch_size=100,callbacks=[tb_hist])

loss_and_metrics=model.evaluate(x_test,y_test, batch_size=32)
print ('loss_and_metrics:'+str(loss_and_metrics))
#保存模型
model.save('vgg16.model')