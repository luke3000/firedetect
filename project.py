import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
#NIST Flashover.mpg.mp4
#ANSUL kitchen fire test.m4v
cap = cv2.VideoCapture('fire1.mp4')
n = 0
take = 0
Simg = np.zeros((256,1))
while(1):
    # Take each frame
    _, img = cap.read()
    frame = img

    if img is None:
        print('shes done')
        break

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # define range of ORANGE color in HSV
    #lower_orange = np.array([5,51,250])
    #upper_orange = np.array([76,255,255])

    lower_orange = np.array([5,51,250])
    upper_orange = np.array([76,255,255])

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
    #maskYCRCB = cv2.GaussianBlur(maskYCRCB,(21,21),0)

    # Bitwise-AND mask and original image
    resHSV = cv2.bitwise_and(frame,frame, mask= maskHSV)
    resYCRCB = cv2.bitwise_and(frame,frame, mask= maskYCRCB)
    resYH = cv2.bitwise_and(frame,hsv,mask= maskYCRCB)
    comb = cv2.bitwise_and(maskHSV,maskYCRCB)

    cv2.imshow('frame',frame)
    cv2.imshow('resHSV',resHSV)
    cv2.imshow('resYCRCB',resYCRCB)
    cv2.imshow('comb', comb)
    #hist = cv2.calcHist( [gray],[0],None,[256],[0,256])
    #cv2.imshow('hist', hist)

    #n is the number of frames in the video
    take += 1
    n += 1
    if take == 1:
        histr = cv2.calcHist([hsv],[1],None,[256],[0,256])

        plt.plot(histr,color = (0,n/1640,1-(n/1640)))
        plt.xlim([0,256])
        take = 0
        Simg = np.append(Simg , histr, axis=1)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
# print(len(histr)) this equals 256
print(n)
Sigma = Simg.std()
Mean = Simg.mean()
limit = Mean+ (3*Sigma) #change from 3*sigma by LT
DimArr=np.shape(Simg)
for i in range(DimArr[0]):
    for j in range(DimArr[1]):
        if Simg[i,j] > limit:
            Simg[i,j] =0
max = max(map(max,Simg))
print (max)
max = max
Simg = Simg/max
Simg = np.multiply(Simg,255)
Gimg = np.zeros((DimArr[0],DimArr[1]))
for i in range(DimArr[0]):
    for j in range(DimArr[1]):
        Gimg[i,j] = Simg[i,j]

Bimg = np.array(Gimg,dtype="uint8")
OUTimg = Image.fromarray(Bimg, mode = 'L')
OUTimg.show()
image_data = np.asarray(OUTimg)
img_colour = cv2.applyColorMap(image_data, cv2.COLORMAP_JET)
cv2.imwrite('BigHist_S.png',img_colour)
cv2.destroyAllWindows()
