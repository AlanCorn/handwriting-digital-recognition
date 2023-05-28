from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
import keras.callbacks
import matplotlib
matplotlib.use('TkAgg')

batch_size=32
num_classes=10

mnist_input_shape = (28,28,1)

(X_train,Y_train),(X_test,Y_test)=mnist.load_data()
X_train=X_train.reshape(-1,28,28,1).astype('float32')/255.0
X_test=X_test.reshape(-1,28,28,1).astype('float32')/255.0
Y_train=np_utils.to_categorical(Y_train)
Y_test=np_utils.to_categorical(Y_test)

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),
                    activation="relu",
                    input_shape=mnist_input_shape))
model.add(Conv2D(16,kernel_size=(3,3),
                    activation="relu"
                    ))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(32,activation="relu"))
model.add(Dense(num_classes,activation='softmax'))



model.compile(loss='categorical_crossentropy',
                optimizer='Adadelta',
                metrics='accuracy')

tb_hist=keras.callbacks.TensorBoard(log_dir='./graph',histogram_freq=0,write_graph=True,write_images=True)
model.fit(X_train,Y_train,epochs=10,batch_size=32,callbacks=[tb_hist])

loss_and_metrics=model.evaluate(X_test,Y_test, batch_size=32)
print ('loss_and_metrics:'+str(loss_and_metrics))
model.save('./test/googlenet.model')