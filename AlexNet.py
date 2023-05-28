# 手写数字识别问题
# AlexNet
import keras.callbacks
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout

(X_train,Y_train),(X_test,Y_test)=mnist.load_data()
X_train=X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')/255.0
X_test=X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')/255.0
Y_train=np_utils.to_categorical(Y_train)
Y_test=np_utils.to_categorical(Y_test)

model = Sequential()
model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=(28, 28, 1), padding='same', activation='relu',
                 kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# 编译网络（定义损失函数、优化器、评估指标）

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics='accuracy')
tb_hist=keras.callbacks.TensorBoard(log_dir='./graph',histogram_freq=0,write_graph=True,write_images=True)
model.fit(X_train,Y_train,epochs=10, batch_size=32, verbose=2,callbacks=[tb_hist])


# 评估:得到损失以及精度
loss_and_metrics=model.evaluate(X_test,Y_test, batch_size=32)
print ('loss_and_metrics:'+str(loss_and_metrics))

model.save('./test/alexnet.model')

