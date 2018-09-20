from sklearn.cross_validation import train_test_split
from keras import layers,models,regularizers
from keras.callbacks import ModelCheckpoint
from keras.metrics import categorical_accuracy
from keras.optimizers import RMSprop
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense , Conv2D , MaxPooling2D , Input , Reshape
from keras.layers import BatchNormalization , Dropout , regularizers , Flatten , Activation , GlobalAveragePooling2D
from keras.optimizers import Adam , Adadelta , RMSprop , SGD
import matplotlib.pyplot as plt
import numpy as np
size=96
train_x = np.load("train.npy")
train_y = np.load("label.npy")
train_y=to_categorical(train_y)
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=0)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
input_data = Input(shape=[size ,size, 3])
conv1 = Conv2D(filters=32 , kernel_size=[3 , 3] , padding='same' , kernel_initializer='he_normal' , use_bias=True , activation='relu')(input_data)
conv1 = BatchNormalization()(conv1)
conv2 = Conv2D(filters=32 , kernel_size=[3 , 3] , padding='same' , kernel_initializer='he_normal' , use_bias=True , activation='relu')(conv1)
conv2 = BatchNormalization()(conv2)
pool1 = MaxPooling2D(pool_size=[2 ,2] , strides=[2 , 2])(conv2)
pool1 = Dropout(0.1)(pool1)

conv3 = Conv2D(filters=64 , kernel_size=[3 , 3] , padding='same' , kernel_initializer='he_normal' , use_bias=True , activation='relu')(pool1)
conv3 = BatchNormalization()(conv3)
conv4 = Conv2D(filters=64 , kernel_size=[3 , 3] , padding='same' , kernel_initializer='he_normal' , use_bias=True , activation='relu')(conv3)
conv4 = BatchNormalization()(conv4)
pool2 = MaxPooling2D(pool_size=[2 , 2] ,strides=[2 , 2])(conv4)
pool2 = Dropout(0.1)(pool2)

conv5 = Conv2D(filters=128 , kernel_size=[3 , 3] , padding='same' , kernel_initializer='he_normal' , use_bias=True , activation='relu')(pool2)
conv5 = BatchNormalization()(conv5)
conv6 = Conv2D(filters=128 , kernel_size=[3 , 3] , padding='same' , kernel_initializer='he_normal' , use_bias=True , activation='relu')(conv5)
conv6 = BatchNormalization()(conv6)
pool3 = GlobalAveragePooling2D()(conv6)

dense1 = Dense(units=128 , activation='relu' , use_bias=True , kernel_initializer='he_normal')(pool3)
dense1 = Dropout(0.1)(dense1)
dense2 = Dense(units=256 , activation='relu' , use_bias=True , kernel_initializer='he_normal')(dense1)
dense2 = Dropout(rate=0.2)(dense2)
dense3 = Dense(units=6 , use_bias=True , kernel_initializer='he_normal')(dense2)
pred = Activation(activation='softmax')(dense3)

model = Model(inputs=input_data , outputs=pred)
model.compile(optimizer=RMSprop(lr=0.001),loss="categorical_crossentropy",metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='yun6.h5',verbose=1, save_best_only=True)
history=model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=25,epochs=25,callbacks=[checkpointer], verbose=1)
history_dict=history.history
loss=history_dict['loss']
val=history_dict['val_loss']
epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label='training loss')
plt.plot(epochs,val,'b',label='val loss')
plt.legend()
plt.show()
score = model.evaluate(X_test, y_test,verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])