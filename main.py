import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2
from tensorflow import keras
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import random
from sklearn.model_selection import train_test_split
#importing the dataset

def import_(path):
    coloums = ['Center','Left','Right','Steering','Throttle','Brake','Speed']
    data = pd.read_csv(os.path.join(path,'driving_log.csv'),names= coloums)
    data['Center'] = data['Center'].apply(Name)
    return data

#modifying the dataset 

def Name(filePath):
    myImagePath = filePath.split('\\')[-1]
    return myImagePath

#Visualzing the dataset and removing the outliers

def Data(data,display=True):
    Bin = 31
    samplesPerBin =  300
    hist, bins = np.histogram(data['Steering'], Bin)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.title('Data Visualisation')
        plt.xlabel('Steering Angle')
        plt.ylabel('No of Samples')
        plt.show()
    removeindexList = []
    for j in range(Bin):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j + 1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeindexList.extend(binDataList)

    print('Removed Images:', len(removeindexList))
    data.drop(data.index[removeindexList], inplace=True)
    print('Remaining Images:', len(data))
    if display:
        hist, _ = np.histogram(data['Steering'], (Bin))
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.title('Balanced Data')
        plt.xlabel('Steering Angle')
        plt.ylabel('No of Samples')
        plt.show()
    return data

#Convering the info of dataset into array to work easier

def ArrayData(path, data):
  imagesPath = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    imagesPath.append( os.path.join(path,'IMG',indexed_data[0]))
    steering.append(float(indexed_data[3]))
  imagesPath = np.asarray(imagesPath)
  steering = np.asarray(steering)
  return imagesPath, steering

#Making more images for our dataset

def augmentImage(imgPath,steering):
    img =  mpimg.imread(imgPath)
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.5, 1.2))
        img = brightness.augment_image(img)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering

#Preparing the image for our model
def preProcess(img):
    img = img[54:120,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

#making the model

def createModel():
  model = keras.models.Sequential()

  model.add(keras.layers.Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
  model.add(keras.layers.Convolution2D(36, (5, 5), (2, 2), activation='elu'))
  model.add(keras.layers.Convolution2D(48, (5, 5), (2, 2), activation='elu'))
  model.add(keras.layers.Convolution2D(64, (3, 3), activation='elu'))
  model.add(keras.layers.Convolution2D(64, (3, 3), activation='elu'))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(100, activation = 'elu'))
  model.add(keras.layers.Dense(50, activation = 'elu'))
  model.add(keras.layers.Dense(10, activation = 'elu'))
  model.add(keras.layers.Dense(1))

  model.compile(keras.optimizers.Adam(learning_rate=0.0001),loss='mse')
  return model

#Batch creation
def dataGen(imagesPath, steeringList, batchSize, train):
    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1)
            if train:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            img = preProcess(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield (np.asarray(imgBatch),np.asarray(steeringBatch))


path = 'data'
data = import_(path)
print('Total Images Imported',data.shape[0])
data = Data(data,display=False)
imagesPath, steerings = ArrayData(path,data)
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings,
                                            test_size=0.2,random_state=5)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))
model = createModel()
history = model.fit(dataGen(xTrain, yTrain, 300, 1),
                                steps_per_epoch=500,
                                epochs=10,
                                validation_data=dataGen(xVal, yVal, 100, 0),
                                validation_steps=50)
model.save('model.h5')
print('Model Saved')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
