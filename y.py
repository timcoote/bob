import numpy as np
import cv2
#from matplotlib import pyplot as plt

#face_cascade = cv2.CascadeClassifier ()
#face_cascade.load('/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier ('/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_eye.xml')

img = cv2.imread('Woody_Allen_0001.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

rjlevels = []
x = []
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
#                levelWeights=x, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)


for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

