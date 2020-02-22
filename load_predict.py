#Loading Saved Model and Doing Predictions 


import tensorflow as tf 
import numpy as np 
import cv2 
import os 








CATEGORY = ["Dog", "Cat"]
img_path = "Image01.jpg"

def preprocess(path): 
	IMG_DIM = 50
	img_arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	new_arr = cv2.resize(img_arr, (IMG_DIM,IMG_DIM))
	tt=  new_arr.reshape(-1, IMG_DIM, IMG_DIM, 1)
	tt = tt/255
	return tt


#Loading TrainedSaved Model
dogVcat_model  = tf.keras.models.load_model('model\DogsVCats.model')


print("Predicting "  + str(img_path) + " ...")
prediction = dogVcat_model.predict([preprocess(img_path)])

deci = float("{0:.2f}".format(prediction[0][0]))
print(str(img_path) + " is a " + CATEGORY[int(round(prediction[0][0]))])
