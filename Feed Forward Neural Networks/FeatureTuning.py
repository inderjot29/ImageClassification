""" 
Created 3rd March, 2017
Author : Inderjot Kaur Ratol
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pylab
from sklearn.preprocessing import OneHotEncoder
import cv2


edgeDetection=False
cropping=False
HOG=False
threshold=500

def UnderSampleTheData(X,Y):
	count=0
	indexes=[]
	previous=None
	for i in range(len(Y)):
		if(previous is None):
			previous=Y[i]
			count=count+1
		elif (previous==Y[i]):
			if(count<threshold):
				count=count+1
			else:
				indexes.append(i)
		elif (previous!=Y[i]):
			count=1
			previous=Y[i]
			
	Y=np.delete(Y,indexes)
	X=np.delete(X,indexes,axis=0)
	return (X,Y)
	
def AugmentTheData(X,Y):
	crop_size=(64,64)
	count=0
	indexesY=[]
	previous=None
	for i in range(len(Y)):
		#if(Y[i]==0 or Y[i]==1):
			#continue
		#else:
			if(previous is None):
				previous=Y[i]
				count=count+1
			elif (previous==Y[i]):
					count=count+1
			elif (previous!=Y[i]):
				if(count<threshold):
					index=i
					arrayX=X.copy()
					previousCount=count
					while count<=threshold:
						img=arrayX[index]
						print index
						if index%2==0:
							new_img=applyRotation(img.transpose(2,1,0),count/float(threshold))
						else:
							new_img=applyApplineTransformations(img.transpose(2,1,0))
						new_img=new_img.transpose(0,1,2)
						new_img=new_img.reshape(1,3,64,64)
						X = np.concatenate((X, new_img),axis=0)
						indexesY.append(Y[i])
						count=count+1
						if(previousCount>0 and index >0):
							index=index-1
							previousCount=previousCount-1
						
				count=1
				previous=Y[i]
			
	Y=np.concatenate((Y,np.asarray(indexesY)),axis=0)
	np.save("trainXAug.npy",X)
	np.save("TrainYAug.npy",Y)
	pylab.hist(Y, 128)
	pylab.show()
	return (X,Y)
	
def applyRotation(img,degree):
	rows,cols,ch = img.shape

	M = cv2.getRotationMatrix2D((cols/2,rows/2),90*degree,1)
	dst = cv2.warpAffine(img,M,(cols,rows))
	return dst
	
def applyApplineTransformations(img):
	rows,cols,ch = img.shape
	
	pts1 = np.float32([[16,16],[32,16],[16,32]])
	pts2 = np.float32([[4,24],[32,16],[16,48]])

	M = cv2.getAffineTransform(pts1,pts2)
	
	dst = cv2.warpAffine(img,M,(cols,rows))
	return dst
	
def applyPerspective(img):
	rows,cols,ch = img.shape

	pts1 = np.float32([[8,8],[56,8],[8,56],[56,56]])
	pts2 = np.float32([[0,0],[56,0],[0,56],[56,56]])

	M = cv2.getPerspectiveTransform(pts1,pts2)

	dst = cv2.warpPerspective(img,M,(50,50))
	return dst

def Transform3dTo1D(inputArray):
	#input array is of shape (number of instances, 3, 64, 64)
	transformed_data=[]
	global HOG,edgeDetection,cropping
	#inputArray -= np.mean(inputArray, axis = 0) # zero-center the data (important)
	#cov = np.dot(inputArray.T, X) / inputArray.shape[0] 
	#U,S,V = np.linalg.svd(cov)
	for img in inputArray:
		img=img.transpose(2,1,0)
		# it does not help in edge detection
		if HOG==True and cropping==True:
			img=applyCropping(img)
		elif edgeDetection==True:
			#return 2D array
			img=applyEdgeDetection(img)
			
		flat=img.flatten()
		transformed_data.append(flat)
		# return 1D array
	return np.array(transformed_data)
	
def applyCropping(img):
	#converts the array into 2D array of shape (3,n*m)
	img=img.reshape(3,-1)
	#img = cv2.resize(img, (60, 60), interpolation = cv2.INTER_CUBIC)
	# crop the image by leaving out 500 enries from both top and bottom.
	#This value is decided after  observing the histograms  of features. 
	#first 500 and last 500 values in the resulting 2D are always 0 in 99% cases.
	# get rid of these.
	crop_img=img[np.ix_(range(3),range(500, 3500))]
	return crop_img
	
def applyEdgeDetection(img):
	#gray scaling and blur is needed to get better edge detection
	gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3, 3), 0)
 
	# apply Canny edge detection using a wide threshold, tight
	# threshold, and automatically determined threshold
	wide = cv2.Canny(blurred, 10, 200)
	tight = cv2.Canny(blurred, 225, 250)
	auto = auto_canny(blurred)
	return auto
	
def applyHOG(img):
	img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
	img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
	img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
	return img
	
def EncodeClassLabels(classLabels):
	enc = OneHotEncoder()
	transformedValues=enc.fit_transform(classLabels).toarray()
	print transformedValues.shape
	return transformedValues
	
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged