import cv2
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

# Load the cascade
face_cascade = cv2.CascadeClassifier(r'E:\\Desktop\\haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread(r'E:\Desktop\facerecog\rohini.jpg')

# BGR to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.3, 6)

# Draw rectangle around the faces 
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0),3)

# Display the output
#cv2.imshow('img', img)
plt.imshow(img)
cv2.waitKey()
cv2.imwrite('E:\\Desktop\\facerecog\\output.face_detection.jpg', img)
