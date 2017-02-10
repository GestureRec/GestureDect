import cv2
import numpy as np
import copy
import densityCompute
import math
from appscript import app

# test
# Environment:
# OS    : Mac OS EL Capitan
# python: 3.5
# opencv: 2.4.13

# parameters
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8 # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 16

# variables
isBgCaptured = 1   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works



# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)






while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (0, 255, 0), 2)
    cv2.imshow('original', frame)

    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        #img = removeBG(frame)
        img = frame[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI

        #cv2.imshow('mask', img)

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blur_hsv = cv2.inRange(hsv, (0, 48, 80), (20, 255, 255))


        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        #blur_hsv= cv2.GaussianBlur(hsv, (blurValue, blurValue), 0)

        cv2.imshow('blur', blur_hsv)
        ret, thresh = cv2.threshold(blur_hsv, threshold, 255, cv2.THRESH_BINARY)

        max=10
        pixelDensity = thresh.copy()

        for col in xrange(thresh.shape[0]):
            for row in xrange(thresh.shape[1]):
                lowY = (col - max) if (col - max) >= 0 else 0
                highY = (col + max) if (col + max) < thresh.shape[0] else thresh.shape[0]

                lowX = (row - max) if (row - max) >= 0 else 0
                highX = (row + max) if (row + max) < thresh.shape[1] else thresh.shape[1]

                for i in xrange(lowY, highY):
                    for j in xrange(lowX, highX):
                        if thresh[i][j] == 1:
                            pixelDensity[i][j] += 1

                for col in xrange(thresh.shape[0]):
                    for row in xrange(thresh.shape[1]):
                        thresh[col][row] = 0

                        if pixelDensity[col][row] > 60:
                            thresh[col][row] = 1





        #--------------------------#


        cv2.imshow('ori', thresh)
        #print thresh.shape[0]


        # get the coutours
        thresh1 = copy.deepcopy(thresh)
        _,contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):  # find the biggest contour (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i
                else:
                    area=cv2.bitwise_not(area)
            res = contours[ci]
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)

            realHandLen = cv2.arcLength(res, True)
            handContour = cv2.approxPolyDP(res, 0.001 * realHandLen, True)
            minX, minY, handWidth, handHeight = cv2.boundingRect(handContour)


            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

            cv2.rectangle(img,(minX,minY),(minX+handWidth,minY+handHeight),(255,0,0))

        cv2.imshow('rec',frame)
        #cv2.imshow('output', drawing)

    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        break
    elif k == ord('t'):
        print 'picture captured!'
        for i in range(50):
            name='palm'+str(i)+'.jpeg'
            cv2.imwrite(name,thresh)
            print i
        print 'finished!'
