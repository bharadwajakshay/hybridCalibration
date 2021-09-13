from numpy.lib.index_tricks import ix_
from utils import createTransformationMatrix
import numpy as np
import sys
import os
import cv2
from utils import *

def selectBestTransform(rT, ptCld, cldColor, image, pointLabels, P, imgCentroid=None, ptCldCentroids=None, mode='photoMetric'):

    scores = np.ones((len(rT),1))*(-10000)
    count = 0
    maxIdx = 0
    for idx,x in enumerate(tqdm(rT)):

        # Decompose the solvePnP results
        rVEc, tVec =  rT[idx]

        R,__ = cv2.Rodrigues(rVEc)
        # create transformation matrix
        transMat = createTransformationMatrix(R, tVec)

        if mode=='photoMetric':
            
            imgPts, FitlCld, filtColor, filtPtCldLabels,__ = backProjectLiDARPtsWColor(ptCld, cldColor, image, pointLabels,  transMat, P)
            if(imgPts.shape[0] > (0.2*ptCld.shape[0])):        
                scores[idx] = photometricLoss(ptCld, imgPts, filtPtCldLabels, image)
                count += 1

        elif mode == 'manhattan':
            #print("Manhattan Distance")
            # Backproject the PointCloud Centroids
            imgCentroidPts, filtCentroidCld = backProjectLiDARPtsCentroids(ptCldCentroids[0], image, transMat, P, filter=False)
            scores[idx] = manhattanLoss(imgCentroid, imgCentroidPts)
            #print('Breakpoint')


        else:
            print('Please select the appropriate lossfunction')


        

    if mode=='photoMetric':
        scores = scores[:count]
        maxIdx = scores.argmax()
    elif mode == 'manhattan':
        maxIdx = scores.argmin()

    print("Max score is: "+str(scores[maxIdx]))

    return(rT[maxIdx],maxIdx)



def getSimilarityKernelScore(labelKernel, label):

    # Assume label Kernel = 3x3 matrix
    kernelSum = 0
    kernelLen = len(labelKernel)

    for idx in range(0,kernelLen):
        if labelKernel[idx] == label:
            kernelSum += 1
        
        # Ignore the padding values
        elif labelKernel[idx] == -1:
            kernelLen += 0

        else:
            kernelSum += 0

    meanKernelSum = kernelSum/kernelLen

    return (meanKernelSum)


def getLabelKernel(pt, image):
    # use -1 
    # check pt = (0,0)

    kernel = np.ones((9,1))*(-1)
    count = 0
    pt = pt.getA1()

    '''
    print('Image Points')
    print(pt)
    '''

    for i in range(pt[0]-1, pt[0]+2):
        for j in range(pt[1]-1, pt[1]+2):
            '''
            print(str(j)+' '+str(i))
            '''
            try:
                kernel[count] = image[j][i]
            except:
                '''
                print('Exception Raised')
                '''
                kernel[count] = -1

            count += 1

    return(kernel)

def getJustTheLabel(pt, image):
    kernel = np.ones((1,1))*(-1)
    pt = pt.getA1()
    try:
        kernel[0] = image[pt[1],pt[0]]
    except:
        kernel[0] =-1
    return(kernel)



def photometricLoss(pointCloud, imgPts, label, image):

    # for each point, take 3x3 kernel in the image and check the label
    # for the same type of label

    imageScore = 0

    for idx in range(0,imgPts.shape[0]):

        #kernel = getLabelKernel(imgPts[idx][0],image)
        kernel = getJustTheLabel(imgPts[idx][0],image)
        ptScore = getSimilarityKernelScore(kernel, label[idx])
        imageScore += ptScore
    
    return(imageScore)


def manhattanLoss(imgPoints, projectedPtCld):

    mDist = np.empty((imgPoints.shape[0],1))
    for idx in range(imgPoints.shape[0]):
        mDist[idx] = np.abs(imgPoints[idx,0]-projectedPtCld[idx,0])+ np.abs(imgPoints[idx,1]-projectedPtCld[idx,1])
    sumDist = np.sum(mDist)

    return(sumDist)

def euclideanLoss(imgPoints, projectedPtCld):
    mDist = np.empty((imgPoints.shape[0],1))
    for idx in range(imgPoints.shape[0]):
        mDist[idx] = np.sqrt(np.power(imgPoints[idx,0]-projectedPtCld[idx,0],2)+ np.power(imgPoints[idx,1]-projectedPtCld[idx,1],2))
    sumDist = np.sum(mDist)

    return(sumDist)




