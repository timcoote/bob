import numpy as np
import cv2
#from matplotlib import pyplot as plt
from glob import glob

#face_cascade = cv2.CascadeClassifier ('/usr/share/OpenCV/haarcascades/haarcascade_smile.xml')
face_cascade = cv2.CascadeClassifier ('/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_eye.xml')
#eye_cascade = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_smile.xml')

def detect (filename):

    print filename
    img = cv2.imread(filename)
    #img = cv2.imread('IMG_20150905_163110.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv3 has newer forms of this method to return more values
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(30,30))

    print len(faces), faces

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
#        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=9, minSize=(150,150))
#        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=2, minSize=(1,1))
#        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=3, minSize=(1,1))
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(1,1))
        print len(eyes), eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    #cv2.imshow('roi',roi_color)
    cv2.waitKey(0)


for f in glob ('data/*.jpg'):
    detect (f)


cv2.waitKey(0)
cv2.destroyAllWindows()

