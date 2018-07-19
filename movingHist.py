import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

cap = cv2.VideoCapture('fire1.mp4')
n = 0
Simg = np.zeros((256,1))
while(1):
    # Take each frame
    _, frame = cap.read()

    if frame is None:
        print('complete')
        break

    # Convert BGR to ycbcr
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    n += 1
    # caluculate histogram of frame. Params: ([name of frame],[chanel 0= y, 1=cb, 2=cr],Mask = None,[scale], [hstsize],[ranges])
    histr = cv2.calcHist([hsv],[0],None,[256],[0,256])
    #append histr to Simg in first axis creating a 256 by number of frames in video
    Simg = np.append(Simg , histr, axis=1)
    #option to exit early, image is still generated up until exit point
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

print(n)
#find standard deviation and mean to then eliminate outlier histograms values because these coresponded to none useful frames
Sigma = Simg.std()
Mean = Simg.mean()
limit = Mean+ (3*Sigma)
#0 values above limit
DimArr=np.shape(Simg)
for i in range(DimArr[0]):
    for j in range(DimArr[1]):
        if Simg[i,j] > limit:
            Simg[i,j] =0
#normalize values from 0 to 255
max = max(map(max,Simg))
print (max)
max = max
Simg = Simg/max
Simg = np.multiply(Simg,255)

#create image using pillow
Bimg = np.array(Simg,dtype="uint8")
OUTimg = Image.fromarray(Bimg, mode = 'L')
image_data = np.asarray(OUTimg)
cv2.imwrite('BigHist_H.png',image_data)
cv2.destroyAllWindows()
