import cv2 as cv
import numpy as np
import itertools # see what this is for
from functools import partial
import threading
from numpy import interp as mapit
import math
import colorsys

low = np.uint8([101, 145, 105])
high = np.uint8([114, 255, 255])

# low = np.uint8([104, 65, 77])
# high = np.uint8([153, 217, 255])

def myround(x, base=1):
    return int(base * round(float(x)/base))

def dist(p1, p2):
	distance = math.sqrt(((p1[0]-p2[0]) * (p1[0]-p2[0])) + ((p1[1]-p2[1]) * (p1[1]-p2[1])))
	return distance

def rand_color(x, y):
	d = dist([0, 0], [x, y])
	# using distance as a metric for hue determination
	hue_value = mapit(d, [0, 1131], [5000, 8000]) % 181
	sat_value = 255
	val_value = 255
	rgb = np.array(colorsys.hsv_to_rgb(hue_value, sat_value, val_value) )
	# gonna use this rgb as bgr :) afterall its random
	# rgb[0] = rgb[2] % 255
	# rgb[1] = rgb[1] % 255
	# rgb[2] = rgb[0] % 255
	return rgb

window = cv.namedWindow('windows7')
canvas = np.zeros((800, 800, 3), np.uint8)
location = np.zeros((200, 200, 20), np.uint8)
pointer_canvas = canvas.copy()
empty = True

capture = cv.VideoCapture(0)
width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))

font = cv.FONT_HERSHEY_COMPLEX
capture.set(cv.CAP_PROP_FPS, 60) # works for videos

fps = int(capture.get(5))
print("fps: ", fps)

ret, prev_frame = capture.read()
prev_frame = cv.flip(prev_frame, flipCode = 1)
prev_frame = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
# prev_frame = cv.adaptiveThreshold(prev_frame, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, 0)
_, prev_frame = cv.threshold(prev_frame, 200, 255, cv.THRESH_BINARY)


while capture.isOpened():
    ret, frame = capture.read()
    pointer_canvas = np.zeros((800, 800, 3), np.uint8)
    img = frame.copy()
    if ret == True:
       # img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
       img = cv.flip(img, flipCode = 1)
       frame_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
       mask = cv.inRange(frame_HSV, low, high)
       tracked = cv.bitwise_or(img, img, mask = mask)

       new_frame = tracked.copy()
       new_frame = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)
       # new_frame = cv.adaptiveThreshold(new_frame, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, 0)
       _, new_frame = cv.threshold(new_frame, 50, 255, cv.THRESH_BINARY)
       diff = cv.bitwise_xor(prev_frame, new_frame)



       mask = cv.erode(mask, None, iterations = 2)
       mask = cv.dilate(mask, None, iterations = 2)
       tracked = img. copy()
       # tracked = cv.bitwise_or(img, img, mask = mask) # any bitwise function can be used here


       contours, hierarchy = cv.findContours(mask.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
       for cont in contours:
           (x, y, w, h) = cv.boundingRect(cont)
           x = myround(x)
           y = myround(y)
           area = cv.contourArea(cont)
           if cv.contourArea(cont) < 400:
               continue
           if h>w:
               cv.rectangle(tracked, (x, y), (x +w, y+h), (0, 255, 255), 3) #drawing a yellow rectangle as the bounding rectangle
               font = cv.FONT_HERSHEY_DUPLEX
               cv.putText(tracked, f'Vertical : {cv.contourArea(cont)}', (15, 15), font, 1, (0, 0, 255))
           else:
               cv.rectangle(tracked, (x, y), (x +w, y+h), (0, 255, 0), 3) #drawing a yellow rectangle as the bounding rectangle
               font = cv.FONT_HERSHEY_DUPLEX
               cv.putText(tracked, f'horizontal: {cv.contourArea(cont)}', (15, 15), font, 1, (0, 0, 255))

           center = [(x + w / 2), (y + h / 2)] # this is the center of the blue part
           # if (h / w) < 0.85 and (h / w) < 1.2:
           if True:
               x = myround(mapit(int(center[0]), [0, width], [0, 800]))
               y = myround(mapit(int(center[1]), [0, height], [0, 800]))
               cv.circle(pointer_canvas, (x, y), 10, rand_color(x, y), thickness = -1)
               if empty or k == ord('r'):
               	empty = True
               	px=x
               	py=y

               if(not empty and k == ord(' ')):
                line_thickness = int(mapit(area, [300, 3500], [3, 25]))
               	if(dist([px, py], [x, y]) > 5):
               		canvas = cv.line(canvas, (px, py), (x, y), [173, 85, 45], thickness = line_thickness)
               	else:
               		canvas = cv.line(canvas, (px, py), (px, py), [173, 85, 45], thickness = line_thickness)
               px = x # put this line in the if to get the joining feature
               py = y # put this line in the if to get the joining feature
               empty = False
           cv.circle(tracked, (int(center[0]), int(center[1])), 15, (255, 255, 255), thickness = -1)
       # cv.drawContours(labimg, contours, -1, (0, 255, 0), 3)
       show_canvas = cv.add(canvas, pointer_canvas)
       cv.imshow('mask', mask)
       cv.imshow('tracked', tracked)
       cv.imshow('windows7', img)
       cv.imshow('canvas', show_canvas)
       cv.imshow('lab', diff)
       cv.imshow('new', new_frame)
       prev_frame = new_frame.copy()
       k = cv.waitKey(1)
       if k == ord('q'):
           break
       pass
capture.release()
cv.destroyAllWindows()
