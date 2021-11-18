import cv2
import numpy as np
from tensorflow.keras.models import load_model


# MODEL
def initializePredectionModel():
	model = load_model("my_custom_model.h5")
	return model


### 1-Preprocessing Image
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




####### 3 - FINDING THE BIGGEST COUNTOUR
def biggestContour(contours):
	biggest = np.array([])
	max_area = 0
	for i in contours:
		area = cv2.contourArea(i) ## DIen tich 1 contour
		if area >50:
			peri = cv2.arcLength(i,True)
			approx = cv2.approxPolyDP(i,0.01 * peri,True)
			if area> max_area and len(approx) == 4:
				biggest = approx
				max_area = area
	return biggest ,max_area 

#### GET PREDECTIONS ON ALL IMAGES
def getPredection(boxes,model):
	result = []
	boxesArray = []
	kernel = np.ones((3, 3), 'uint8')
	for image in boxes:
		## Prepare image
		# img = np.asarray(image)
		# img = img[4:img.shape[0]-4,6:img.shape[1]-6]
		
		# img = cv2.erode(img, kernel, iterations=2)
		
		# img = cv2.resize(img,(28,28))
		# imgBlur = cv2.GaussianBlur(img,(3,3),0)
		# img = cv2.addWeighted(img,2,imgBlur,-1,0)
		# img = cv2.equalizeHist(img)
	






		# img = np.asarray(image)
		# img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]
		# img = cv2.resize(img, (28, 28))
		# boxesArray.append(img)
		# img = img / 255
		# img = img.reshape(1, 28, 28, 1)

		



		# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		img = np.asarray(image)
		img = img[6:img.shape[0]-6,6:img.shape[1]-6]
		img = cv2.resize(img,(32,32))
		boxesArray.append(img)
		Threshold = 100
		img [img < Threshold] = 0
		img [img > Threshold] = 255
		img = cv2.equalizeHist(img)
		img = img/255   

		img = img .reshape(1,32,32,1)










		predictions = model.predict(img)
		# classIndex = model.predict_classes(img)
		classIndex = np.argmax(predictions,axis =-1)
		probabilityValue = np.amax(predictions)
		# print(classIndex,probabilityValue)

		# SAVE TO RESULT
		if probabilityValue > 0.6:
			result.append(classIndex[0])
		else:
			result.append(0)
	return result,boxesArray










#### SPLIT THE IMAGE INTO 81 DIFFERENT IMGAES
def splitBoxes(img):
	rows = np.vsplit(img,9)
	boxes = []
	for r in rows:
		cols = np.hsplit(r,9)
		for box in cols:
			boxes.append(box)
	return boxes 

###### DISPLAY THE SOLUTION ON THE IMAGE

def displayNumbers(img,numbers,color =(0,255,0)):
	secW = int(img.shape[1]/9)
	secH = int(img.shape[0]/9)
	for x in range(0,9):
		for y in range(0,9):
			if numbers [(y*9)+x] !=0:
				cv2.putText(img,str(numbers[(y*9)+x]),(x*secW+int(secW/2)-10,int((y+0.8)*secH)),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,color,2,cv2.LINE_AA)
	return img






# 6- Stack all the images in one window
def stackImages(imgArray,scale):
	rows = len(imgArray)
	cols = len(imgArray[0])
	rowsAvailable = isinstance(imgArray[0],list)
	width = imgArray[0][0].shape[1]
	height = imgArray[0][0].shape[0]
	if rowsAvailable:
		for x in range(0,rows):
			for y in range(0,cols):
				imgArray[x][y] = cv2.resize(imgArray[x][y],(0,0),None,scale,scale)
				if len(imgArray[x][y].shape)==2 :
					imgArray[x][y] = cv2.cvtColor(imgArray[x][y],cv2.COLOR_GRAY2BGR)
		imageBlank = np.zeros((height,width,3),np.uint8)
		hor = [imageBlank] *rows
		hor_con = [imageBlank]*rows
		for x in range (0,rows):
			hor[x] = np.hstack(imgArray[x])
			hor_con[x] = np.concatenate(imgArray[x])
		ver = np.vstack(hor)
	else:
		for x in range(0,rows):
			imgArray[x] = cv2.resize(imgArray[x],(0,0),None,scale,scale)
			if len(imgArray[x].shape) ==2 :
				imgArray[x] = cv2.cvtColor(imgArray[x],cv2.COLOR_GRAY2BGR)
		hor = np.hstack (imgArray)
		ver = hor
	return ver


# 7 DRAW GRID
def drawGrid(img):
	secW = int(img.shape[1]/9)
	secH = int(img.shape[0]/9)
	for i in range(0,9):
		point1 = (0,secH*i)
		point2 = (img.shape[1],secH*i)
		point3 = (secW*i,0)
		point4 = (secW*i,img.shape[0])
		cv2.line(img,point1,point2,(255,255,0),2)
		cv2.line(img,point3,point4,(255,255,0),2)
	return img


def OverwriteImage(img,imgoverwrite):

	imgGray = cv2.cvtColor(imgoverwrite, cv2.COLOR_BGR2GRAY)
	_, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV) 
	imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
	imgFinal = cv2.bitwise_and(img, imgInv)
	imgFinal = cv2.bitwise_or(imgFinal, imgoverwrite)
	return imgFinal