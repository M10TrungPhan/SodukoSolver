print("Setting Up")
import os 
import cv2
import numpy as np


heightImg = 450
widthImg = 450


pathImage="Sudoku-Medium.jpg"   ### File anh


def reorder (myPoints):
	myPoints = myPoints.reshape((4,2))
	myPointsNew = np.zeros((4,1,2),dtype=np.int32)
	add = myPoints.sum(1)
	myPointsNew[0] = myPoints[np.argmin(add)]
	myPointsNew[3] = myPoints[np.argmax(add)]
	diff = np.diff(myPoints,axis =1)
	myPointsNew[1] = myPoints[np.argmin(diff)]
	myPointsNew[2] = myPoints[np.argmax(diff)]
	return myPointsNew

def preProcess(img):
	imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
	imgThreshold = cv2.adaptiveThreshold(imgBlur, 255,1,1,11,2)
	return imgThreshold

#### 2- REORDER POINT
def reorder (myPoints):
	myPoints = myPoints.reshape((4,2))
	myPointsNew = np.zeros((4,1,2),dtype=np.int32)
	add = myPoints.sum(1)
	myPointsNew[0] = myPoints[np.argmin(add)]
	myPointsNew[3] = myPoints[np.argmax(add)]
	diff = np.diff(myPoints,axis =1)
	myPointsNew[1] = myPoints[np.argmin(diff)]
	myPointsNew[2] = myPoints[np.argmax(diff)]
	return myPointsNew

def splitBoxes(img):
	rows = np.vsplit(img,9)
	boxes = []
	for r in rows:
		cols = np.hsplit(r,9)
		for box in cols:
			boxes.append(box)
	return boxes 



####### 3 - FINDING THE BIGGEST COUNTOUR
def biggestContour(contours):
	biggest = np.array([])
	max_area = 0
	for i in contours:
		area = cv2.contourArea(i) ## DIen tich 1 contour
		if area >50:
			peri = cv2.arcLength(i,True)
			approx = cv2.approxPolyDP(i,0.02 * peri,True)
			if area> max_area and len(approx) == 4:
				biggest = approx
				max_area = area
	return biggest ,max_area 


#### 1- Prepare the image
img= cv2.imread(pathImage)
print(img.shape)
img = cv2.resize(img,(widthImg,heightImg))
imgBlank = np.zeros((heightImg,widthImg,3),np.uint8)
imgThreshold = preProcess(img)
cv2.imshow('Display Image', imgThreshold)


# #### 2- FIND ALL COUNTOURS
imgContours = img.copy()
imgBigContour = img.copy()
contours, hierachy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(imgContours,contours,-1,(0,255,0),3)

#### 3- FIND THE BIGGEST COUNTOUR AND USE IT AS SUDOKU

biggest, maxArea = biggestContour(contours)
# print(biggest) 

biggest =reorder(biggest)
cv2.drawContours(imgBigContour,biggest,-1,(0,0,255),25)
pts1 = np.float32(biggest)
pts2 = np.float32(([0,0],[heightImg,0],[0,widthImg],[heightImg,widthImg]))
matrix = cv2.getPerspectiveTransform(pts1,pts2) ## Tao ma tran bien cac pts1 => pts2

imgWarpColored = cv2.warpPerspective(img,matrix,(heightImg,widthImg))
imgWarpColored = cv2.detailEnhance(imgWarpColored, sigma_s=10, sigma_r=0.15)

imgDetectedDigits = imgBlank.copy()
imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

#### 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
imgSolveDigits = imgBlank.copy()
boxes = splitBoxes(imgWarpColored)
i =0
b = []
kernel = np.ones((5, 5), 'uint8')


for image in boxes:
		## Prepare image
		
	a = str(i) +".jpg"	
	img = np.asarray(image)
	img = img[6:img.shape[0]-6,6:img.shape[1]-6]
	imgBlur = cv2.GaussianBlur(img,(5,5),0)
	img = cv2.addWeighted(img,2,imgBlur,-1,0)
	img = cv2.erode(img, kernel, iterations=1)
	img = cv2.resize(img,(28,28))
	# img = cv2.equalizeHist(img)
	b.append(img)
	i = i+1
cv2.imshow("abc",b[2])
print(b[2])
cv2.waitKey(0)

