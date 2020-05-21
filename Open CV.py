#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2

 
imgName = "lena_color_512.tif"
 
image= cv2.imread(imgName)
cv2.imshow('MyImage', image)
 
 
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[7]:


import cv2
import matplotlib.pyplot as plt
 
 
 
image = cv2.imread("lena_color_512.tif")
plt.imshow(image, cmap = 'gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([]) # for hading ticks in the x and y positions
plt.show()


# In[11]:


import cv2
import numpy as np
import cv2
 
 
 
# Drawing Shapes
 
def ImageProcessing():
    image = np.zeros((512, 512, 3), np.uint8)
 
    cv2.line(image, (456,123), (200,20), (0,0,255),5)
    cv2.rectangle(image, (200,60), (20,200), (255,0,0), 3)
    cv2.circle(image, (80,80), 50, (0,255,0), 4)
 
    mytext = "Hello World"
 
    cv2.putText(image, mytext, (100,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255) )
 
    cv2.imshow('Black Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
 
 
 
ImageProcessing()


# In[14]:


import cv2
 
 
 
 
image = cv2.imread("lena_color_256.tif")
 
 
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
 
 
faces = face_cascade.detectMultiScale(image, scaleFactor = 1.1, minNeighbors = 7)
 
print("Faces Detected", len(faces))
 
 
for x,y,w,h in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 4)
 
 
cv2.imshow("Face Detected", image)
 
 
 
cv2.waitKey(0)
cv2.destroyAllWindows()

