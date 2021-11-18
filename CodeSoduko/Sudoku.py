print("Setting Up")
import os 
import cv2
import numpy as np
from utils import *
import Sudoku_Solver

heightImg = 450
widthImg = 450
pathImage="7.jpg"
# pathImage="3.png"

model = initializePredectionModel() # LOAD MODEL


img= cv2.imread(pathImage)

img = cv2.resize(img,(widthImg,heightImg))
imgBlank = np.zeros((heightImg,widthImg,3),np.uint8)
imgThreshold = preProcess(img)
# cv2.imwrite("Ảnh nhị phân.jpg",imgThreshold)

imgContours = img.copy()
imgBigContour = img.copy()
contours, hierachy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(imgContours,contours,-1,(0,255,0),3)


biggest, maxArea = biggestContour(contours)

if biggest.size != 0:
	biggest =reorder(biggest)
	cv2.drawContours(imgBigContour,biggest,-1,(0,0,255),25)
	# cv2.imwrite("BigContour.jpg",imgBigContour)
	pts1 = np.float32(biggest)
	pts2 = np.float32(([0,0],[heightImg,0],[0,widthImg],[heightImg,widthImg]))
	matrix = cv2.getPerspectiveTransform(pts1,pts2) ## Tao ma tran bien cac pts1 => pts2
	
	imgWarpColored = cv2.warpPerspective(img,matrix,(heightImg,widthImg))
	# cv2.imwrite("SodukoCrop.jpg",imgWarpColored)
	
	imgDetectedDigits = imgBlank.copy()
	imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
	# cv2.imwrite("SodukoCrop1.jpg",imgWarpColored)
	
#### 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
	imgSolveDigits = imgBlank.copy()
	boxes = splitBoxes(imgWarpColored)

	
	numbers,boxesArray = getPredection(boxes,model)

	imgDetectedDigits = displayNumbers(imgDetectedDigits,numbers,color =(255,0,255))
	numbers = np.asarray(numbers)
	posArray = np.where(numbers>0,0,1)
	cv2.imwrite("abc.jpg",imgWarpColored)
	# ######## 5 -FIND THE SOLUTIONS OF THE BOARD
	board = np.array_split(numbers,9)

	print("Sudoku is sloving")
	try:
		Sudoku_Solver.solve(board)
		print("Congratulation. Sudoku is sloved ")
	except:
		pass
		print("Khong giai duoc")

	flatList =[]
	
	for sublist in board:
		for item in sublist:
			flatList.append(item)

	# print(np.array((flatList)).reshape(9,9))
	solvedNumbers = flatList*posArray
	imgSolveDigits = displayNumbers(imgSolveDigits,solvedNumbers,color =(0,0,255))

	imgAllNumber = imgBlank.copy()
	imgAllNumber = displayNumbers(imgAllNumber,flatList,color =(0,0,255))


	############ 6- OVERPLAY SOLUTION

	pts2 = np.float32(biggest) 
	pts1 = np.float32(([0,0],[heightImg,0],[0,widthImg],[heightImg,widthImg]))
	matrix = cv2.getPerspectiveTransform(pts1,pts2)
	imgInvWarpColored = img.copy()
	imgInvWarpColored = cv2.warpPerspective(imgSolveDigits,matrix,(widthImg,heightImg))
	
	# inv_perspective = cv2.addWeighted(imgInvWarpColored,1,img,0.5,1)

	
	# imgGray = cv2.cvtColor(imgInvWarpColored, cv2.COLOR_BGR2GRAY)
	# # cv2.imwrite("imgGray.jpg",imgGray)
	# _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV) 
	# # cv2.imwrite("imgInv.jpg",imgInv)
	# imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
	# # cv2.imwrite("imgInv1.jpg",imgInv)
	# imgFinal1 = cv2.bitwise_and(img, imgInv)
	# # cv2.imwrite("imgAnd.jpg",imgFinal)
	# imgFinal = cv2.bitwise_or(imgFinal1, imgInvWarpColored)
	# # # cv2.imwrite("imgFinal.jpg",imgFinal)

	imgFinal = OverwriteImage(img,imgInvWarpColored)



	
	








	# imgDetectedDigits = drawGrid(imgDetectedDigits)
	# imgSolveDigits =drawGrid(imgSolveDigits)

	imageArray =([img, imgThreshold,imgContours,imgBigContour,imgWarpColored],[imgDetectedDigits,imgAllNumber,imgSolveDigits ,imgInvWarpColored,imgFinal])
	# imageArray = ([imgInvWarpColored,imgGray,imgInv,imgFinal1,imgFinal])
	# imageResult = ([img,imgFinal])
	
	# img1 = img.copy()
	# imgArray =([img,imgFinal])
	

	stackImage = stackImages(imageArray,0.5)
	# cv2.imwrite("image.jpg",stackImage)

	cv2.imshow("Process",stackImage)
	imgArray =([img,imgFinal])
	
	stackImage = stackImages(imgArray,1)
	cv2.imshow("Final",stackImage)

else: 
	print("No Sudoku")

cv2.waitKey(0)
