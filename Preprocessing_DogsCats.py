
import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import os
import pickle
import random

DATADIR = "PetImages/"
CATEGORIES = ["Dog","Cat"]

tr_data = []

IMG_DIM = 50

def create_tr_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_nm = CATEGORIES.index(category)
        for img in os.listdir(path): 
            try : 
                img_arr = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_arr = cv2.resize(img_arr, (IMG_DIM,IMG_DIM))
                tr_data.append([new_arr, class_nm])
            except Exception as e : 
                pass
            
            
create_tr_data() 
random.shuffle(tr_data)

X = [] #Features
y = [] #Labels



for features, labels in tr_data: 
	X.append(features)
	y.append(labels)

X = np.array(X).reshape(-1,IMG_DIM,IMG_DIM,1)

pickle_out = open("Xs.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()



pickle_out = open("ys.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()


#opening pickle data pickle_in = open('Xs.pickle',wb) -> Xs = pickle.load(pickle_in) 