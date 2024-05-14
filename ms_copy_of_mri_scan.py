# -*- coding: utf-8 -*-
"""ms copy of mri_scan.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1iWYKi3tqKmMCEvx9__doYoV4xkoMaQdM
"""

# pip install pydicom

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
import datetime
import time
import math as m
import pydicom
import glob

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

pms=[]
image_path = glob.glob('/content/drive/MyDrive/ParkinCareDataset/parkinson scan/PPMI*.dcm')
for path in image_path:
    try:
        dataset = pydicom.dcmread(path)
        pms.append(dataset.pixel_array)
    except Exception as e:
        print(f"Error reading DICOM file {path}: {str(e)}")

print("Total number of pixel arrays:", len(pms))

import os
import pydicom
from PIL import Image
import cv2
import numpy as np

# Function to convert DICOM to JPEG using PIL
def dicom_to_jpeg_pil(dicom_file, output_folder):
    # Read DICOM file
    ds = pydicom.dcmread(dicom_file)

    # Convert pixel data to numpy array
    pixel_array_numpy = ds.pixel_array

    # Normalize pixel values to 8-bit range (0-255)
    pixel_array_normalized = ((pixel_array_numpy.astype(np.float32) / np.max(pixel_array_numpy)) * 255).astype(np.uint8)

    # Convert numpy array to grayscale image
    img = Image.fromarray(pixel_array_normalized)

    # Convert numpy array to grayscale image
    #img = Image.fromarray(pixel_array_numpy)

    # Save as JPEG
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(dicom_file))[0] + ".jpg")
    img.save(output_file)

    return output_file

# Function to convert DICOM to JPEG using OpenCV
def dicom_to_jpeg_opencv(dicom_file, output_folder):
    # Read DICOM file
    ds = pydicom.dcmread(dicom_file)

    # Convert pixel data to numpy array
    pixel_array_numpy = ds.pixel_array

    # Save as JPEG using OpenCV
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(dicom_file))[0] + ".jpg")
    cv2.imwrite(output_file, pixel_array_numpy)

    return output_file

import os

def get_files_in_folder(parkinson_path):
  dicom_file = []
  for filename in os.listdir(parkinson_path):
    dicom_file.append(filename)
  return dicom_file


parkinson_path = "/content/drive/MyDrive/ParkinCareDataset/parkinson scan"
file_list = get_files_in_folder(parkinson_path)
print(len(file_list))



#output_folder = "DCM"
# Example usage:
#dicom_file = "/content/drive/MyDrive/MRI Dataset/PD/PPMI*.dcm"
output_folder = "/content/drive/MyDrive/ParkinCareDataset/PD1"

# Convert using PIL
for dicom_file in file_list:  # Change dicom_files_list to file_list
    jpeg_file_pil = dicom_to_jpeg_pil(os.path.join(parkinson_path, dicom_file), output_folder)  # Use the full path to the DICOM file
    print("Saved JPEG using PIL:", jpeg_file_pil)

# Convert using OpenCV
for dicom_file in file_list:  # Change dicom_files_list to file_list
    jpeg_file_opencv = dicom_to_jpeg_opencv(os.path.join(parkinson_path, dicom_file), output_folder)  # Use the full path to the DICOM file
    print("Saved JPEG using OpenCV:", jpeg_file_opencv)

#dataset preparartion class to make it easier to load the data
class DataSet:

    def __init__(me,location,categories,resize=True,
                 lheight=500,lwidth=500,grayscale=True,shuffled=False,
                 apply=None,count=1000,multiclass=False,enhance=False):
        me.categories=categories
        me.datadir=location
        me.lheight=lheight
        me.lwidth=lwidth
        me.grayscale=grayscale
        me.shuffled=shuffled
        me.multiclass=multiclass
        me.apply=apply
        me.count=count
        me.enhance=enhance
        me.dataset=me.create_traindata()
        if resize==True:
            me.dataset=me.resizeIt(me.dataset)




    def resizeIt(me,traindata_array):
        resized_traindata=[]
        resized_traindata_temp=[]
        for img in traindata_array[0]:

            new_image_array=cv2.resize(img,(me.lheight,me.lwidth))
            resized_traindata_temp.append(np.array(new_image_array))
        array=[np.array(resized_traindata_temp),np.array(traindata_array[1])]
        return(array)



    def create_traindata(me):
        traindata=[]
        for cats in me.categories:
            n=0
            print("cats{}".format(cats))
            path=os.path.join(me.datadir,cats)
            class_num=me.categories.index(cats)
            for img in os.listdir(path):
                if(me.grayscale==True and me.enhance==True):
                    print("hello")
                    y=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

                    y=cv2.resize(y,(512,512))


                    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5,5))
                    img_array = clahe.apply(y)

                    img_array = cv2.GaussianBlur(y,(3,3),1)


                    n=n+1
                    print(str(n)+" images loaded successfully",end='')
                    if n>=me.count:
                      break

                elif(me.enhance==True):
                    print("path:{} img:{}".format(path,img))
                    img_array=cv2.imread(os.path.join(path,img))
                    if img_array is None:
                      print(f"Failed to read image: {os.path.join(path, img)}")
                    img_array=cv2.resize(img_array,(512,512))
                    print("here")
                    img_yuv_1 = cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)


                    img_yuv = cv2.cvtColor(img_yuv_1,cv2.COLOR_RGB2YUV)

                    y,u,v = cv2.split(img_yuv)



                    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5,5))
                    y = clahe.apply(y)

                    y = cv2.GaussianBlur(y,(3,3),1)

                    img_array_1 = cv2.merge((y,u,v))
                    img_array = cv2.cvtColor(img_array_1,cv2.COLOR_YUV2RGB)

                    n=n+1
                    print(str(n)+" images loaded successfully",end='')
                    if n>=me.count:
                      break
                else:
                    img_array=cv2.imread(os.path.join(path,img))

                    n=n+1
                    print(str(n)+" images loaded successfully",end='')
                    if n>=me.count:
                      break
                if(me.multiclass==False):
                    traindata.append([img_array,class_num])
                else:
                    traindata.append([img_array,me.classes(class_num=class_num,classes=len(me.categories))])
            print(len(traindata))
            print()

        if(me.shuffled==True):
          random.shuffle(traindata)
          print("shuffled")
        traindata_img=[]
        traindata_lab=[]
        for sets in traindata:
            traindata_img.append(sets[0])
            traindata_lab.append(sets[1])
        traindata=[traindata_img,traindata_lab]
        return(traindata)

    def classes(me,class_num,classes):
        array = [0 for i in range(classes)]
        array[class_num]=1
        return(array)

#path of the folder containing subfolder with images
path="/content/drive/MyDrive/ParkinCareDataset"

#names of the subfolders
class_names = ['healthy','PD1']

#function to load the dataset into the variable dataset
dataset=DataSet(path,categories=class_names,lheight=512,lwidth=512,grayscale=False,apply=None,count=3000,shuffled=True,multiclass=True,enhance=True)

# Create DataSet object
#dataset = DataSet(path, categories=class_names, resize=True, lheight=512, lwidth=512, grayscale=False, shuffled=True, multiclass=True, enhance=True)

# Data contains the numpy image array
data = dataset.dataset
#this returns a shuffled numpy array with the format [[images][labels]] to data
#if data is not None and not data.size == 0:
    #data = cv2.resize(img_array, (512, 512))
    # Continue processing the resized image
#else:
    #print("Error: Empty or invalid image array encountered.")

print(type(dataset))
print(len(data))
print(len(data[1]))
print(data[0].shape)
print(len(data[1][:20]))

#x=len(data[0])
#test_sample_size=int(0.001*x)
#train_sample_size=x-test_sample_size

#splitting the data into training set and test set,with test_sample_size being the percentage of total dataset for test set
#(tr_img,tr_lab),(te_img,te_lab)=(data[0][:train_sample_size],data[1][:train_sample_size]),(data[0][train_sample_size:],data[1][train_sample_size:])

from sklearn.model_selection import train_test_split

# Assuming data is a tuple containing features and labels
X = data[0]
y = data[1]

# Splitting the data into 70% training and 30% test sets with randomness
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)  # You can set a specific random state for reproducibility

# Printing the sizes of the resulting sets
print("Training set size:", len(X_train))
print("Test set size:", len(X_test))

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#print(tr_lab)
#print(te_lab)
#plt.imshow(tr_img[18],cmap='gray')
#plt.show()
#print(tr_lab[0])

#tr_img = tr_img.reshape(-1,512,512,3)

#defining our model ,the description of the model is provided separately
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
lesion_Classifier=Sequential()
lesion_Classifier.add(Convolution2D(16,(3,3),input_shape=(512,512,3),activation='relu'))
lesion_Classifier.add(MaxPooling2D(pool_size=(2,2)))
lesion_Classifier.add(Convolution2D(32,(3,3),activation='relu'))
lesion_Classifier.add(MaxPooling2D(pool_size=(2,2)))
lesion_Classifier.add(Convolution2D(64,(3,3),activation='relu'))
lesion_Classifier.add(MaxPooling2D(pool_size=(2,2)))
lesion_Classifier.add(Convolution2D(64,(3,3),activation='relu'))
lesion_Classifier.add(MaxPooling2D(pool_size=(2,2)))
lesion_Classifier.add(Convolution2D(128,(3,3),activation='relu'))
lesion_Classifier.add(MaxPooling2D(pool_size=(2,2)))
lesion_Classifier.add(Convolution2D(128,(3,3),activation='relu'))
lesion_Classifier.add(MaxPooling2D(pool_size=(2,2)))

lesion_Classifier.add(Flatten())
lesion_Classifier.add(Dense(512,activation='relu'))
lesion_Classifier.add(Dense(256,activation='relu'))
lesion_Classifier.add(Dense(2,activation='softmax'))

lesion_Classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#create a tensorboard callback for our model's training,with log dir being the location of the logs
log_dir = os.path.join(
    "logs",
    "fit",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#name for the tensorboard logs
name="images_ewith-sn-1".format(int(time.time()))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

#training the model for 25 epochs with a validation set ,10% of the training set,and mapping the progress to tensorboard
history = lesion_Classifier.fit(X_train, y_train, epochs=50, validation_split=0.3, callbacks=[tensorboard_callback])

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#saving the weights to h5 format
lesion_Classifier.save("/content/drive/MyDrive/Saved Model/parkinson first stage99.h5")
#model.save("model_vgg16_mypd_acc90.h5")

"""**PRETRAINED VGG16**

old
"""

#from sklearn import metrics
#predictions = lesion_Classifier.predict_classes(data[0])
# Predict class probabilities
class_probabilities = lesion_Classifier.predict(data[0])

# Extract predicted classes (assuming classes are one-hot encoded)
predictions = np.argmax(class_probabilities, axis=1)
label = [int(np.where(r==1)[0][0]) for r in data[1]]

print(label)
print(predictions)

#report = lesion_Classifier.classification_report(label,predictions)
#confusion = lesion_Classifier.confusion_matrix(label,predictions,labels=[0,1])
from sklearn.metrics import classification_report, confusion_matrix
# Compute classification report
report = classification_report(label, predictions)

# Compute confusion matrix
confusion = confusion_matrix(label, predictions, labels=[0, 1])

print(report)
print(confusion)

print(lesion_Classifier.summary())

"""**Testing**"""

img_array=cv2.imread('/content/drive/MyDrive/ParkinCareDataset/Test/no 2.jpg')
img_array=cv2.resize(img_array,(512,512))

img_yuv_1 = cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)
plt.imshow(img_yuv_1)
plt.show()

img_yuv = cv2.cvtColor(img_yuv_1,cv2.COLOR_RGB2YUV)

y,u,v = cv2.split(img_yuv)



clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5,5))
y = clahe.apply(y)

y = cv2.GaussianBlur(y,(3,3),1)

img_array_1 = cv2.merge((y,u,v))
img_array = cv2.cvtColor(img_array_1,cv2.COLOR_YUV2RGB)
plt.imshow(img_array)
plt.show()
test_im = img_array.reshape(-1,512,512,3)

#print(class_names[lesion_Classifier.predict_classes(test_im)[0]])
class_probabilities = lesion_Classifier.predict(test_im)
#print(class_probabilities)
# Extract the predicted class index
predicted_class_index = np.argmax(class_probabilities, axis=1)[0]

# Print the predicted class name
print("Predicted class:", class_names[predicted_class_index])
if class_names[predicted_class_index] == 'healthy':
    print("You do not have Parkinson's Disease")
else:
    print("You have Parkinson's Disease")
anp = img_array[230:200+130,200:200+110]
plt.imshow(anp)

