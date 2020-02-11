import os
import numpy as np
import cv2 # OpenCV
from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import accuracy_score
from joblib import dump, load

nbins = 9 # broj binova
cell_size = (6, 6) # broj piksela po celiji
block_size = (3, 3) # broj celija po bloku
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))

AM ='.'+ os.path.sep+'car_data'+os.path.sep+'car_data'+os.path.sep+'train'+os.path.sep+'Aston Martin'+os.path.sep
Audi ='.'+ os.path.sep+'car_data'+os.path.sep+'car_data'+os.path.sep+'train'+os.path.sep+'Audi'+os.path.sep
BMW ='.'+ os.path.sep+'car_data'+os.path.sep+'car_data'+os.path.sep+'train'+os.path.sep+'BMW'+os.path.sep
Ferari ='.'+ os.path.sep+'car_data'+os.path.sep+'car_data'+os.path.sep+'train'+os.path.sep+'Ferari'+os.path.sep


model =load('cars.joblib')
if model == None:
    train_X =[]
    labels=[]
    for glavni in os.listdir(AM):
            img = cv2.cvtColor(cv2.imread(AM+glavni), cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (150, 150), interpolation = cv2.INTER_AREA)
            hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
            train_X.append(hog.compute(img))
            labels.append('AM')
    for glavni in os.listdir(Audi):
            img = cv2.cvtColor(cv2.imread(Audi+glavni), cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (150, 150), interpolation = cv2.INTER_AREA)
            hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
            train_X.append(hog.compute(img))
            labels.append('Audi')
    for glavni in os.listdir(BMW):
            img = cv2.cvtColor(cv2.imread(BMW+glavni), cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (150, 150), interpolation = cv2.INTER_AREA)
            hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
            train_X.append(hog.compute(img))
            labels.append('BMW')
    for glavni in os.listdir(Ferari):
            img = cv2.cvtColor(cv2.imread(Ferari+glavni), cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (150, 150), interpolation = cv2.INTER_AREA)
            hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
            train_X.append(hog.compute(img))
            labels.append('Ferari')
            
    x = np.array(train_X)
    y = np.array(labels)
    x_train = reshape_data(x)
    print('Train shape: ', x.shape,y.shape)
    clf_svm = SVC(kernel='linear') 
    clf_svm.fit(x_train, y)
    y_train_pred = clf_svm.predict(x_train)
    print("Train accuracy: ", accuracy_score(y, y_train_pred))
    dump(clf_svm, 'cars.joblib')

#Testiranje
test ='.'+ os.path.sep+'test'+os.path.sep

for glavni in os.listdir(test):
            train_X=[]
            img = cv2.cvtColor(cv2.imread(test+glavni), cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (150, 150), interpolation = cv2.INTER_AREA)
            hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
            train_X.append(hog.compute(img))
            x = np.array(train_X)        
            x_train = reshape_data(x) 
            print(glavni,model.predict(x_train)[0])

