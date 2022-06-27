import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from numpy import argmax
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import load_model
import pickle

testratio = 0.2  # tester sur 20% des images
validratio = 0.2  # 20%du reste des images pour la validation
imageDim = (32, 32, 3)

# ----- importer les images---------
path = './AIData/myData/myData'
labels = './AIData/labels2.csv'
compt = 0
images = []
classe = []

liste = os.listdir(path)
nbrclasses = len(liste)
print("le nombre total des classes est :", nbrclasses)
print("importez classes...")
for x in range(0, len(liste)):
    piclist = os.listdir(path + '/' + str(compt))
    for y in piclist:
        Img = cv2.imread(path + '/' + str(compt) + '/' + y)
        images.append(Img)
        classe.append(compt)
    print(compt, end=" ")
    compt += 1
images = np.array(images)
classe = np.array(classe)
# ----------diviser les donnees---------------
X_train, X_test, y_train, y_test = train_test_split(images, classe, test_size=testratio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validratio)
# X_train est un tableau d image a entrainer
# y_train id de la classe coresspond

# vérification si le nombre d'images correspond au nombre de labels pour chaque donnée
print(" data shapes ")
print("train ", end=" ")
print(X_train.shape, y_train.shape)
print("validation ", end=" ")
print(X_validation.shape, y_validation.shape)
print("test ", end=" ")
print(X_test.shape, y_test.shape)
assert (X_train.shape[0] == y_train.shape[0]), "Le nombre d'images n'est pas égal au nombre de labels dans training set."
assert (X_validation.shape[0] == y_validation.shape[0]), "Le nombre d'images n'est pas égal au nombre de labels en validation set "
assert (X_test.shape[0] == y_test.shape[0]), "Le nombre d'images n'est pas égal au nombre de labels en test set "
assert (any(X_train.shape[1:]) == any(imageDim)), "la dimension des image du train est mauvais"
assert (any(X_validation.shape[1:]) == any(imageDim)), "la dimension des image du validation est mauvais"
assert (any(X_test.shape[1:]) == any(imageDim)), "la dimension des image du test est mauvais"
# lire le fichier csv.
data = pd.read_csv(labels)
print("data shape", data.shape, type(data))
print(data.columns)
print(data.head())


#        traitement d'images :
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def equalizeHist(img):
    img = cv2.equalizeHist(img)
    return img


def processing(img):
    img = grayscale(img)
    img = equalizeHist(img)
    img = img / 255  # NORMALISER LES VALEURS ENTRE 0 ET 1 AU LIEU DE 0 À 255.
    return img


X_train = np.array(list(map(processing, X_train)))  # TO IRETATE AND PREPROCESS ALL IMAGES
X_validation = np.array(list(map(processing, X_validation)))
X_test = np.array(list(map(processing, X_test)))
# cv2.imshow("GrayScale Images",X_train[random.randint(0,len(X_train)-1)])
cv2.imshow("Grayscal",X_train[random.randint(0,len(X_train)-1)])
# reformer les images
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
print(X_train)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
# augmentation des images
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, rotation_range=10,
                             shear_range=0.1)
datagen.fit(X_train)
batches = datagen.flow(X_train, y_train, batch_size=50)
X_batches, y_batches = next(batches)
#  POUR MONTRER DES ÉCHANTILLONS D'IMAGES AUGMENTÉES
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()
for i in range(15):
    axs[i].imshow(X_batches[i].reshape(imageDim[0], imageDim[1]))
    axs[i].axis('off')
plt.show()
print(y_train)
print(y_train.shape)
y_train=to_categorical(y_train,nbrclasses)
y_validation=to_categorical(y_validation,nbrclasses)
y_test=to_categorical(y_test,nbrclasses)
print(y_train)
print(y_train.shape)
print(y_validation)
print(y_validation.shape)
print(y_test)
print(y_test.shape)
inverted=argmax(y_train[0])
print(inverted)
def mymodel():
    nbr_filter=60
    size_of_filter=(5,5)
    size_of_filter2=(3,3)
    size_of_pool=(2,2)
    nbr_of_node=500
    model=Sequential()
    model.add(Conv2D(nbr_filter,size_of_filter,activation='relu',input_shape=(imageDim[0],imageDim[1],1))) #AJOUTER PLUS DE COUCHES DE CONVOLUTION = MOINS DE CARACTÉRISTIQUES MAIS PEUT AUGMENTER LA PRÉCISION.
    model.add(Conv2D(nbr_filter,size_of_filter,activation='relu'))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Conv2D(nbr_filter//2,size_of_filter2,activation='relu'))
    model.add(Conv2D(nbr_filter // 2, size_of_filter2, activation='relu'))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(nbr_of_node,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nbrclasses,activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
model=mymodel()
print(model.summary())
history=model.fit(datagen.flow(X_train,y_train,batch_size=50),epochs=10,validation_data=(X_validation,y_validation),shuffle=1)
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])
model.save('model_trained.h5')










