#Training/Testing the Neural Net
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

import numpy as np 




#Loading Features

picklex_in = open('Xs.pickle',"rb")
X = pickle.load(picklex_in) 

#Loading Labels
pickley_in = open('ys.pickle',"rb")
y = pickle.load(pickley_in) 

#Normalize 
X = X/255
y = np.array(y)


model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))



model.compile(loss="binary_crossentropy",
	optimizer="adam",
	metrics=['accuracy'])

model.fit(X, y, batch_size=25, epochs=6, validation_split=0.1)

#Saving our model
model.save('model\DogsVCats.model') 


#Loading the Model
#tf.keras.models.load_model('saved_model/my_model')