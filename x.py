import bob.io.base
import bob.io.image
import bob.ip.color
from pkg_resources import resource_filename
from os.path import join
import cv2

def get_data (f):
#    return (bob.io.base.load (resource_filename ('bob.ip.flandmark', join ('data', f))))
    return (bob.io.base.load (f))


pic = get_data ('IMG_20150905_163110.jpg')
#pic = get_data ('lena.jpg')
pic_grey = bob.ip.color.rgb_to_gray(pic)
#pic_grey = bob.ip.color.rgb_to_gray(bob.io.base.load('face.jpg'))

try:
    from cv2 import CascadeClassifier
    cc = CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
    face_bbxs = cc.detectMultiScale(pic_grey, 1.3, 4, 0, (20, 20))
except ImportError: #if you don't have OpenCV, do it otherwise
    face_bbxs = [[214, 202, 183, 183]] #e.g., manually
print(face_bbxs)

x, y, width, height = face_bbxs [0]

from bob.ip.flandmark import Flandmark

localiser = Flandmark ()

keypoints = localiser.locate (pic_grey, y, x, height, width)

print keypoints

from matplotlib import pyplot
from bob.ip.draw import box, cross, plus, line

def show_diff (a, b):
    print (a == b).all ()
    if (a == b).all ():
        print "arrays are the same"
    else:
        print (a - b).nonzero ()
        #print [i for i in prebox - pic]

prebox = pic.copy ()

show_diff (prebox, pic)

box (pic, (y, x), (height, width), (255, 0, 0))

#print len (keypoints), dir (keypoints [-1:][0]), keypoints [-1:][0].shape

for k in keypoints [:-1]:
    cross (pic, k.astype (int), 5, (255,0,0))
    print k.astype (int)

show_diff (prebox, pic)

plus (pic, keypoints[-1:][0].astype (int), 6, (0,0, 255))

#print keypoints [1].astype (int), keypoints [2].astype (int)

#cv2.line (pic, keypoints[1].astype (int), keypoints[2].astype (int), (255,255,255), 10)
#cv2.line (pic, (651, 340), (632, 476), (255,0,0), 10)

preline = pic.copy ()
cv2.line (pic, (0, 0), (1, 1), (255,0,0), 1)
print "cv2 line"
show_diff (preline, pic)

pic1 = preline.copy ()
line (pic1, (0,0), (1,1), (255,0,0))
print "bob"
show_diff (preline, pic1)

print "delta"
show_diff (pic, pic1)


#line (pic, (340, 651), (476, 632), (255,0,0))
line (pic, (651, 340), (632, 476), (255,0,0))

show_diff (pic, preline)

pyplot.imshow (pic.transpose (1,2,0))
pyplot.show ()

#print pic.__class__
#print dir (pic), pic.shape
