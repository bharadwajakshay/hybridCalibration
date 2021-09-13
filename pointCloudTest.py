from typing import Sequence
import numpy as np
import os
import sys
import open3d as o3d
import yaml
from scipy import spatial
from utils import *
from dataLoader import *
import glob


from segmentImg import processImage
from ptCldSegmentation import processPoint
from visualize import *
from photometricLoss import *

_Debug = True

'''
Nr.     Sequence name     Start   End
---------------------------------------
00: 2011_10_03_drive_0027 000000 004540
01: 2011_10_03_drive_0042 000000 001100
02: 2011_10_03_drive_0034 000000 004660
03: 2011_09_26_drive_0067 000000 000800
04: 2011_09_30_drive_0016 000000 000270
05: 2011_09_30_drive_0018 000000 002760
06: 2011_09_30_drive_0020 000000 001100
07: 2011_09_30_drive_0027 000000 001100
08: 2011_09_30_drive_0028 001100 005170
09: 2011_09_30_drive_0033 000000 001590
10: 2011_09_30_drive_0034 000000 001200
'''

corrR09_26 = np.mat([[9.998817e-01, 1.511453e-02, -2.841595e-03],[-1.511724e-02, 9.998853e-01, -9.338510e-04],[2.827154e-03, 9.766976e-04, 9.999955e-01]])
camMat09_26 = np.mat([[9.597910e+02, 0.000000e+00, 6.960217e+02], [0.000000e+00, 9.569251e+02, 2.241806e+02],[0.000000e+00, 0.000000e+00, 1.000000e+00]])  
distMat09_26 = np.mat([-3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02])
rT09_26 = np.mat([[7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],[1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02], [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],[0, 0, 0, 1]]) 
p09_26 = np.mat([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01], [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01], [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]])

corrR09_30 = np.mat([[9.999019e-01, 1.307921e-02, -5.015634e-03],[-1.307809e-02, 9.999144e-01, 2.561203e-04],[5.018555e-03, -1.905003e-04, 9.999874e-01]])
camMat09_30 = np.mat([[9.591977e+02, 0.000000e+00, 6.944383e+02], [0.000000e+00, 9.529324e+02, 2.416793e+02], [0.000000e+00, 0.000000e+00, 1.000000e+00]])
distMat09_30 = np.mat([-3.725637e-01, 1.979803e-01, 1.799970e-04,1.250593e-03, -6.608481e-02])
rT09_30 = np.mat([[7.027555e-03, -9.999753e-01, 2.599616e-05, -7.137748e-03], [-2.254837e-03 ,-4.184312e-05, -9.999975e-01, -7.482656e-02], [9.999728e-01, 7.027479e-03, -2.255075e-03, -3.336324e-01],[0, 0, 0, 1]])
p09_30 = np.mat([[7.070912e+02, 0.000000e+00, 6.018873e+02, 4.688783e+01],[0.000000e+00, 7.070912e+02, 1.831104e+02, 1.178601e-01],[0.000000e+00, 0.000000e+00, 1.000000e+00, 6.203223e-03]])

corrR10_03 = np.mat([[9.999191e-01, 1.228161e-02, -3.316013e-03], [-1.228209e-02, 9.999246e-01, -1.245511e-04],[3.314233e-03, 1.652686e-04, 9.999945e-01]])
camMat10_03 = np.mat([[9.601149e+02, 0.000000e+00, 6.947923e+02], [0.000000e+00, 9.548911e+02, 2.403547e+02], [0.000000e+00, 0.000000e+00, 1.000000e+00]])
distMat10_03 = np.mat([-3.685917e-01, 1.928022e-01, 4.069233e-04, 7.247536e-04, -6.276909e-02])
rT10_03 = np.mat([[7.967514e-03, -9.999679e-01, -8.462264e-04, -1.377769e-02],[-2.771053e-03, 8.241710e-04, -9.999958e-01, -5.542117e-02],[9.999644e-01, 7.969825e-03, -2.764397e-03, -2.918589e-01],[0, 0, 0, 1]])
p10_03 = np.mat([[7.188560e+02, 0.000000e+00, 6.071928e+02, 4.538225e+01],[0.000000e+00, 7.188560e+02, 1.852157e+02, -1.130887e-01],[0.000000e+00, 0.000000e+00, 1.000000e+00, 3.779761e-03]])


def readvelodynepointcloud(path_to_file):
    pointcloud = np.fromfile(path_to_file, dtype=np.float32).reshape(-1, 4)
    intensity_data = pointcloud[:,3]
    
    # Return points ignoring the reflectivity 
    return(pointcloud)


def main():

    pointCloudDir = "/home/akshay/5TB-HDD/Datasets/kitti-semantic/dataset/sequences/07/velodyne"
    labelDir = "/home/akshay/5TB-HDD/Datasets/kitti-semantic/dataset/sequences/07/labels"
    imageDir =  "/home/akshay/5TB-HDD/Datasets/kitti-semantic/dataset/sequences/07/image_3"
    yamlFile = "/home/akshay/programming/pytorchvenv/Cylinder3D/config/label_mapping/semantic-kitti.yaml"

    if int(pointCloudDir.split('/')[-2])<3:
        rectR = corrR10_03
        camMat = camMat10_03
        distMat = distMat10_03
        ProjMat = p10_03
        rT = rT10_03
        #rT = np.eye(4)
    elif int(pointCloudDir.split('/')[-2])>3:
        rectR = corrR09_30
        camMat = camMat09_30
        distMat = distMat09_30
        ProjMat = p09_30
        rT = rT09_30
        #rT = np.eye(4)
    else:
        rectR = corrR09_26
        camMat = camMat09_26
        distMat = distMat09_26
        ProjMat = p09_26
        rT = rT09_26
        #rT = np.eye(4)
    

    for filename in glob.glob(pointCloudDir+'/*.bin'):
        # Debug
        # Override filename
        filename='/home/akshay/5TB-HDD/Datasets/kitti-semantic/dataset/sequences/07/velodyne/001024.bin'

        seqID = filename.split('/')[-1].split('.')[0]     
        labelFile = os.path.join(labelDir,seqID+'.label')
        imageFile = os.path.join(imageDir,seqID+'.png')

        # Process images 1st
        img, imageLabel,imgCentroids, origImg = processImage(imageFile, seqID, _Debug )

        with open(yamlFile, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        colorMap = semkittiyaml['color_map']



        ptCld, labels, cldColor,  ptCldCentroids = processPoint(filename, labelFile, colorMap, seqID, _Debug)

        # Correct point clouds for camera two 
        rectR = createTransformationMatrix(rectR,np.zeros((3,1)))
        rectfiedPtCld = correctForTransformation(ptCld, rectR)[:,:3]

        imgPts, FitlCld, filtColor, filtPtCldLabels, correctedPointCld = backProjectLiDARPtsWColor(ptCld, cldColor, img, labels,  rT, ProjMat)
        ptCldCentroidsExt = extractCentroids(ptCldCentroids)
        imgCentroidPts, filtCentroidCld = backProjectLiDARPtsCentroids(ptCldCentroidsExt, img, rT, ProjMat)
        
        """
        ################################       Debug       ########################################
        filtCentroidCld = filtCentroidCld[:,:3]
        
        filteredCentroidRearrange = np.ones_like((filtCentroidCld))
        filteredCentroidRearrange[0] = filtCentroidCld[1]
        filteredCentroidRearrange[1] = filtCentroidCld[2]
        filteredCentroidRearrange[2] = filtCentroidCld[0]
        filteredCentroidRearrange[3] = filtCentroidCld[3]

        points2D = imgCentroids[1][0]

        # Calculate thrPNP problem
        status, rVec, tVec = cv2.solvePnP(np.array(filteredCentroidRearrange,dtype='float32'), np.array(points2D,dtype='float32'), camMat, distMat,flags=cv2.SOLVEPNP_IPPE)
        R,__ = cv2.Rodrigues(rVec)

        # create transformation matrix
        transMat = createTransformationMatrix(R, tVec)

        imgPts, FitlCld, filtColor, filtPtCldLabels,correctedPointCld = backProjectLiDARPtsWColor(correctedPointCld, cldColor, img, labels,  transMat, ProjMat)

        # visualize hte pointcloud
        displayPointsOnImage(imgPts,filtColor, img,'.Debug/images/Depth_img_Overlap.png')
        displayPointsOnImage(imgPts,filtColor, origImg,'.Debug/images/Depth_color_img_Overlap.png',segment=False)


        score = photometricLoss(FitlCld, imgPts, filtPtCldLabels, imageLabel)
        print(score)

        print("Image Centroid")
        print(imgCentroids)
        print("Point cloud projected Centroid")
        print(imgCentroidPts)
        print("Point cloud Centroid")
        print(filtCentroidCld)

        # visualize hte pointcloud
        displayPointsOnImage(imgPts,filtColor, img, '.Debug/images/Depth_withCentroid_img_Overlap.png', imgCentroidPts)
        displayPointsOnImage(imgPts,filtColor, origImg,'.Debug/images/Depth_color_img_Overlap.png',centroid=imgCentroidPts,segment=False)
        ##############################################################################################
        """
        displayPointsOnImage(imgPts,filtColor, img, '.Debug/images/Depth_withCentroid_img_Overlap.png', imgCentroidPts)
        displayPointsOnImage(imgPts,filtColor, origImg, '.Debug/images/Depth_withCentroid_img_Overlap_O.png', imgCentroidPts, segment=False)
        
        score = photometricLoss(FitlCld, imgPts, filtPtCldLabels, imageLabel)
        print(score)

                
        rT, imgCentroids,centroids = getRigidBodyTransformation(imgCentroids, ptCldCentroids, camMat, distMat)

        bestRT,centroidIdx = selectBestTransform(rT, ptCld, cldColor, imageLabel, labels, ProjMat, imgCentroid=imgCentroids, ptCldCentroids=centroids, mode='manhattan')

        coarserVEc, coarsetVec =  bestRT
        coarseR,__ = cv2.Rodrigues(coarserVEc)
        # create transformation matrix
        coarseTransMat = createTransformationMatrix(coarseR, coarsetVec)

        print(coarseTransMat)

        imgPts, FitlCld, filtColor, filtPtCldLabels,__ = backProjectLiDARPtsWColor(ptCld, cldColor, img, labels,  coarseTransMat, ProjMat)
        imgCentroidPts, filtCentroidCld = backProjectLiDARPtsCentroids(centroids[centroidIdx], img, coarseTransMat, ProjMat)

        # visualize hte pointcloud
        displayPointsOnImage(imgPts,filtColor, img,'.Debug/images/Depth_img_Overlap.png',centroid=imgCentroidPts)
        displayPointsOnImage(imgPts,filtColor, origImg,'.Debug/images/Depth_color_img_Overlap.png',centroid=imgCentroidPts,segment=False)

        #refine The initial projection
        coarsetVecTest = np.zeros_like(coarsetVec)
        fineRVec, fineTVec = refineTranformation(imgCentroids,centroids[centroidIdx], coarserVEc, coarsetVecTest, camMat, distMat)

        fineR,__ = cv2.Rodrigues(fineRVec)
        # create transformation matrix
        fineTransMat = createTransformationMatrix(fineR, fineTVec)

        print(fineTransMat)

        imgPts, FitlCld, filtColor, filtPtCldLabels,__ = backProjectLiDARPtsWColor(ptCld, cldColor, img, labels,  fineTransMat, ProjMat)
        imgCentroidPts, filtCentroidCld = backProjectLiDARPtsCentroids(centroids[centroidIdx], img, fineTransMat, ProjMat)

        # visualize hte pointcloud
        displayPointsOnImage(imgPts,filtColor, img,'.Debug/images/Depth_img_Overlap_Refined.png',centroid=imgCentroidPts)
        displayPointsOnImage(imgPts,filtColor, origImg,'.Debug/images/Depth_color_img_Overlap_Refined.png',centroid=imgCentroidPts,segment=False)

        
        # Plot all possible transformations
        for idx in range(len(rT)):
            coarserVEc, coarsetVec =  rT[idx]
            coarseR,__ = cv2.Rodrigues(coarserVEc)
            # create transformation matrix
            coarseTransMat = createTransformationMatrix(coarseR, coarsetVec)

            print(coarseTransMat)

            imgPts, FitlCld, filtColor, filtPtCldLabels,__ = backProjectLiDARPtsWColor(ptCld, cldColor, img, labels,  coarseTransMat, ProjMat)
            imgCentroidPts, filtCentroidCld = backProjectLiDARPtsCentroids(centroids[idx], img, coarseTransMat, ProjMat)

            # visualize hte pointcloud
            displayPointsOnImage(imgPts,filtColor, img,'.Debug/images/Depth_img_Overlap_'+str(idx)+'.png',centroid=imgCentroidPts)   
            displayPointsOnImage(imgPts,filtColor, origImg,'.Debug/images/Depth_color_img_Overlap_'+str(idx)+'.png',centroid=imgCentroidPts,segment=False)
        



        
        print(filename)
        print(labelFile)
        
    
if __name__ == "__main__":
    main()