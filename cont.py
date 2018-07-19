import cv2
import numpy as np

from PIL import Image
import imutils
import argparse

cap = cv2.VideoCapture('fire1.mp4')
Simg = np.zeros((256,1))
MIN_THRESH = 0
while(1):
    # Take each frame
    _, img = cap.read()
    frame = img

    if img is None:
        print('shes done')
        break

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of ORANGE color in HSV
    #lower_orange = np.array([5,51,250])
    #upper_orange = np.array([76,255,255])

    lower_orange = np.array([5,20,240])
    upper_orange = np.array([28,255,255])

    # Convert BGR to YCrCb
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    split = cv2.split(ycrcb)
    y = np.mean(split[0])
    cr = np.mean(split[1])
    cb = np.mean(split[2])
    # define range of ORANGE color in YCrCb
    lower_orange_y = np.array([y,0,cb])
    upper_orange_y = np.array([255,cr,255])

    # Threshold the HSV and YCrCb image to get only blue colors
    maskHSV = cv2.inRange(hsv, lower_orange, upper_orange)
    #maskHSV = cv2.GaussianBlur(maskHSV,(21,21),0)
    maskYCRCB = cv2.inRange(ycrcb, lower_orange_y, upper_orange_y)
    #ret,thresh = cv2.threshold(maskYCRCB,127,255,0, cv2.THRESH_BINARY)
    #maskYCRCB = cv2.GaussianBlur(maskYCRCB,(21,21),0)

    # Bitwise-AND mask and original image
    resHSV = cv2.bitwise_and(frame,frame, mask= maskHSV)
    resYCRCB = cv2.bitwise_and(frame,frame, mask= maskYCRCB)
    #resh = cv2.bitwise_not(maskHSV)

    blurred = cv2.GaussianBlur(maskHSV,(21,21),0)
    maskYCRCB = cv2.GaussianBlur(maskYCRCB,(21,21),0)
    #cv2.drawContours(blurred, contours, -1, (255,0,0), 3)

# find contours in the thresholded image

    _,cnts,_ = cv2.findContours(blurred.copy(), cv2.RETR_EXTERNAL,
	   cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    conv = cv2.convexHull(cnts)

# loop over the contours
    for c in cnts:
    	# compute the center of the contour
    	if cv2.contourArea(c) > MIN_THRESH:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.drawContours(frame,cnts, c, -1, (255, 0, 150), 2)
            #epsilon = 0.1*cv2.arcLength(c,True)
            #approxCurve = cv2.approxPolyDP(c,epsilon,True)
            cv2.fillPoly(maskHSV, pts =[c], color=(255,255,255))
            cv2.fillPoly(resHSV, pts =[c], color=(255,255,255))
            cv2.circle(frame, (cX, cY), 7, (255, 0, 150), -1)
            cv2.putText(frame, "fire!", (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #distance = (cx- c[0])^2 + (cy-c[1])^2
            #cv2.imshow("Image", comb)

    comb = cv2.bitwise_and(maskHSV,maskYCRCB)
    cv2.imshow('frame',frame)
    cv2.imshow('resHSV',resHSV)
    cv2.imshow('resYCRCB',resYCRCB)
    cv2.imshow('comb', comb)


    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
# print(len(histr)) this equals 256
cv2.destroyAllWindows()
