from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
import imutils
import cv2
import numpy as np


def izmeniMnist(mnist_data):

    for i in range (len(mnist_data)):
        
        slika = mnist_data[i].reshape(28,28)
        slika = (slika).astype('uint8')
        slika = exposure.rescale_intensity(slika, out_range=(0, 255))

        #kernel = np.ones((2,2), np.uint8)
        #slika = cv2.dilate(slika,kernel)    
        
        im2, contours, hierarchy = cv2.findContours(slika, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cords = cv2.boundingRect(contours[0])
        #cv2.rectangle(slika, (cords[0], cords[1]) ,(cords[0] + cords[2], cords[1] + cords[3]), (0,0,255), 2)
        slika_cropped = slika[cords[1] : cords[1] + cords[3], cords[0] : cords[0] + cords[2]]
        #slika_erode = cv2.erode(slika_cropped, kernel)
        slika_resized = cv2.resize(slika_cropped, (28, 28)) 

        #if i == 5:
        #    cv2.imshow('prvobitna slicica', slika)
         #   cv2.imshow('cropovana slicica', slika_cropped)
         #   cv2.imshow('resizovana slicica', slika_resized)
          #  cv2.waitKey(0)   
        
        temp = slika_resized.flatten()
        mnist_data[i] = temp 
      

    return mnist_data


def getKNN():

    mnist = fetch_mldata('MNIST original')
    data   = izmeniMnist(mnist.data) 
    labels = mnist.target.astype('int')

    
    """
    train_rank = 5000
    test_rank = 100
    #------- MNIST subset --------------------------
    train_subset = np.random.choice(data.shape[0], train_rank)
    test_subset = np.random.choice(data.shape[0], test_rank)

    # train dataset
    train_data = data[train_subset]
    train_data = editData(train_data)
    train_labels = labels[train_subset]

    # test dataset
    test_data = data[test_subset]
    test_data = editData(test_data)
    test_labels = labels[test_subset]
    """
    knn = KNeighborsClassifier(n_neighbors=1, algorithm='brute').fit(data, labels)

    return knn
