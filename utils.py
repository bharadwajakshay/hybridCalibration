from numpy.lib.type_check import imag
import open3d as o3d
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as rot
from PIL import Image
import cv2
import itertools
from tqdm import tqdm

from torchvision import transforms

def getCentroid(nparray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(nparray)
    centroid = pcd.get_center()
    return centroid

def getAllCentroids(points):
    redIdx = np.empty(0)
    initialPoint = 0
    centroid = np.empty(0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Filter out point farther than 3 meters
    pcd.remove_statistical_outlier(1,std_ratio=0.5)

    pcdKDTree = o3d.geometry.KDTreeFlann(pcd)

    while(pcd.has_points()):
        [k,idx,__] = pcdKDTree.search_radius_vector_3d(pcd.points[initialPoint], 5)
        if not(k<15):
            centroid = np.append(centroid,pcd.select_by_index(idx).get_center())
        pcd = pcd.select_by_index(idx,invert=True)
        pcdKDTree = o3d.geometry.KDTreeFlann(pcd)
    
    centroid = centroid.reshape(-1,3)
    
    return centroid

def normalizePtCld(pointcloud):
    centroid = np.mean(pointcloud, axis=0)
    pointcloud = pointcloud - centroid
    m = np.max(np.sqrt(np.sum(pointcloud**2, axis=1)))
    pointcloud = pointcloud / m
    return pointcloud

def readimgfromfileCV2(path_to_file):
    image = cv2.imread(path_to_file)
    return([image.shape[1],image.shape[0],image])

def readimgfromfilePIL(path_to_file):
    image = Image.open(path_to_file)
    return(image.width,image.height,image)

def readvelodynepointcloud(path_to_file):
    '''
    The velodyne data that is presented is in the form of a np array written as binary data.
    So to read the file, we use the inbuilt fuction form the np array to rad from the file
    '''

    pointcloud = np.fromfile(path_to_file, dtype=np.float32).reshape(-1, 4)

    # Return points ignoring the reflectivity 
    return(pointcloud)

def resizeImgForResNet(image):
    # Since resnet is trainted to take in 224x224, we want resize the image of the shortest size to be 224
    imageScaleRatio = 224/375
    newImW = int(image.shape[1]*imageScaleRatio)
    newImH = int(image.shape[0]*imageScaleRatio)
    image = cv2.resize(image,(newImW,newImH))
    return (image)
     

def normalizePILGrayImg(image):
    # Convert the image into float
    image = np.array(image)
    image = image.astype('float32')
    for idx in range(0,image.shape[2]):
        max  = image[:,:,idx].max().max()
        image[:,:,idx] = np.divide(image[:,:,idx],max)
    
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    return (image)

def normalizePILImg(image):
    # Convert the image into float
    image = np.array(image)
    image = image.astype('float32')
    image = np.divide(image,255)
    
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    return (image)

def plotSpheres(centroids):
    spheres = [] 
    eye = np.eye(4)
    for idx in range(0,centroids.shape[0]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        sphere.compute_vertex_normals()
        transform = eye
        transform[:3,3] = centroids[idx].transpose()
        sphere.transform(transform)
        sphere.paint_uniform_color([0.1, 0.8, 0.1])
        #o3d.visualization.draw_geometries([sphere,pcd])
        spheres.append(sphere)

    return(spheres)

def filterPointCloud(pointcloud, label, distance=50, negativeZ=True):
    """
    filterIdx = np.where(pointcloud<50)
    pointcloud = pointcloud[filterIdx[0],:3]
    label = label[filterIdx[0],:]

    if negativeZ:
        filterZ = np.where(pointcloud[:,0]>0)
        pointcloud = pointcloud[filterZ[0],:3]
        label = label[filterZ[0],:]
    """

    condition = (pointcloud[:,0]<50) #& (pointcloud[:,0]>0)
    pointcloud = pointcloud[condition]
    label = label[condition]

    return(pointcloud, label)

def plotPtCldNCentroids(points, color,centroids,seqId):
    
    eye =np.eye(4)

    # draw centrods
    spheres = plotSpheres(centroids)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(color)
    #o3d.visualization.draw_geometries([spheres[0],spheres[1],spheres[2],spheres[3],spheres[4],spheres[5],spheres[6],spheres[7],spheres[8],spheres[]carspcd])
    #o3d.visualization.draw_geometries([carspcd])
    plotList = spheres
    plotList.append(pcd)
    o3d.visualization.draw_geometries(plotList)


def getRigidBodyTransformation(imageCentroids, pointCloudCentroids, K, D):

    # Get contents of image centroids, class wise

    bicycleCentroidNo = len(imageCentroids[0][0])
    carCentroidNo = len(imageCentroids[1][0])
    motorcycleCentroidNo = len(imageCentroids[2][0])
    peopleCentroidno = len(imageCentroids[3][0])

    totalImgCentroids =  bicycleCentroidNo+carCentroidNo+motorcycleCentroidNo+peopleCentroidno

    if(totalImgCentroids<3):
        return(-1)

    count = 0

    # get all permutation of point cloud centroids
    allpermofbicycle = list(itertools.permutations(pointCloudCentroids[0][0]))
    allpermofcar = list(itertools.permutations(pointCloudCentroids[1][0],totalImgCentroids))
    allpermofmotorcycle = list(itertools.permutations(pointCloudCentroids[2][0],totalImgCentroids))
    allpermofpeople = list(itertools.permutations(pointCloudCentroids[3][0]))

    retVal = []
    centroid = []
    imageCentroid = np.array(imageCentroids[1][0], dtype='float32')

    for ittr in range(0,len(allpermofcar)):
        points3D = np.array(allpermofcar[ittr][:(totalImgCentroids)], dtype='float32')
        #flipPoints2D = flipImgRowsNCols(np.array(imageCentroids[1][0], dtype='float32'))
        points2D = np.array(imageCentroids[1][0], dtype='float32')

        #status, rVec, tVec, __ = cv2.solvePnPRansac(points3D, points2D, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        status, rVec, tVec, __ = cv2.solvePnPRansac(points3D, points2D, K, None, flags=cv2.SOLVEPNP_EPNP)
        
        #status, rVec, tVec = cv2.solvePnP(points3D, points2D, K, None,flags=cv2.SOLVEPNP_AP3P )
        if status:
            # from the setup we know that the x,y,z is not more than 1m
            # hence filter out every T > 1
        
            if not (np.abs(tVec[0])>2 or np.abs(tVec[1])>2 or np.abs(tVec[2])>2):
                retVal.append([rVec,tVec])
                centroid.append(points3D)
        

    return(retVal,imageCentroid,centroid)
     


def backProjectLiDARPtsWColor(pointCloud, color, image, labels, rT, P):
    '''
    step 1: Correct point cloud for the transformation matrix
    '''
    rTCorrectedPtCld = correctForTransformation(pointCloud, rT)

    '''
    Step 2: Use Projection matrix to project the point cloud to images 
    '''
    imgCld = projectptstoimgplane(rTCorrectedPtCld,P)

    '''
    Step 3: Filter the point clouds in the range of the image
    '''
    filtImgCld, filtPtCld, filtColor, filtLabels = filterPtsInImgPlane(imgCld, rTCorrectedPtCld, color, labels, image.shape[1], image.shape[0]) 

    return(filtImgCld, filtPtCld, filtColor, filtLabels,rTCorrectedPtCld)


def backProjectLiDARPtsCentroids(centroids, image, rT, P, filter=True):

    '''
    step 1: Correct point cloud for the transformation matrix
    '''
    rTCorrectedPtCld = correctForTransformation(centroids, rT)

    '''
    Step 2: Use Projection matrix to project the point cloud to images 
    '''
    imgCld = projectptstoimgplane(rTCorrectedPtCld,P)

    if not filter:
        return(imgCld, rTCorrectedPtCld)

    '''
    Step 3: Filter the point clouds in the range of the image
    '''
    filtImgCld, filtPtCld = filterPtsInImgPlane(imgCld, rTCorrectedPtCld, None, None, image.shape[1], image.shape[0],centroid=True) 

    return(filtImgCld, filtPtCld)




def convertToHomogeneousCoordinates(ptcloud):
    
    ptcloud = np.hstack((ptcloud[:,:3],np.ones((ptcloud.shape[0],1))))
    return(ptcloud)

def correctForTransformation(pointCloud, rT):
    '''
    Expected transformation matrix 4x4
    rT = r00 r01 r02 t0
         r10 r11 r12 t1
         r20 r21 r22 t2
         0   0   0   1
    '''

    # Convert point clouds to Homogeneous coordnates
    pointCldHomo = convertToHomogeneousCoordinates(pointCloud)
    pointCloud = np.matmul(rT,np.transpose(pointCldHomo))

    # Convert the point cloud from 4xN to Nx4
    pointCloud = np.transpose(pointCloud)

    return(pointCloud)


def projectptstoimgplane(pointCloud, P):

    '''
    Expected projection matrix 3X4
    rT = r00 r01 r02 t0
         r10 r11 r12 t1
         r20 r21 r22 t2
    '''
    pointCloudHomo = convertToHomogeneousCoordinates(pointCloud)

    pointCloudHomo = np.transpose(pointCloudHomo)

    # Project the points on to 2D image space [u,v,1]
    imgPts = np.matmul(P,pointCloudHomo)

    # Divide the u,v by the resedual to obtain the right access
    imgPts = np.transpose(imgPts)
    imgPts[:,0] /= imgPts[:,2]
    imgPts[:,1] /= imgPts[:,2]
    
    imgPts = imgPts.astype(int)

    return(imgPts[:,:2])

def filterPtsInImgPlane(points2D, points3D, color, labels, imgWidth, imgHeight, img = None, centroid=False):

    conditionXImg = (points2D[:,0] < imgWidth) & (points2D[:,0] >= 0)
    conditionYImg = (points2D[:,1] < imgHeight) & (points2D[:,1] >= 0)
    conditionZPtCld = (points3D[:,2] > 0)
    conditionZPtCld = np.reshape(conditionZPtCld,(conditionZPtCld.shape[0],1))
    condition = conditionXImg & conditionYImg  &  conditionZPtCld
    
    idx = np.where(condition)
    
    pointsImg = points2D[idx[0],:]
    pointsCld = points3D[idx[0],:]
    if centroid:
        return [pointsImg, pointsCld]
    color = color[idx[0],:]
    labels = labels[idx[0],:]

    return [pointsImg, pointsCld, color, labels]

def generateDepthImage(pointsImg, pointsCld, imgWidth, imgHeight, img=None):

    minDist = 0
    maxDist = np.amax(pointsCld[:,2])
    intensityMax = 255
    intensityMin = 0

    # scaletointensity (value - minDist)/(maxDist-minDist) x (intensityMax-intensityMin) + intensityMin

    scaletointensity = (((pointsCld[:,2] - minDist)/(maxDist-minDist))*(intensityMax-intensityMin)) + intensityMin
    
    # assert depth map as intensty values
    pointsImg = np.hstack((pointsImg,scaletointensity.reshape(scaletointensity.shape[0],1)))
    
    # convert points to int
    pointsImg = pointsImg.astype(int)

    # Create a new image
    depthimg = np.zeros((imgHeight,imgWidth),dtype=np.uint8)
    depthimg[pointsImg[:,1],pointsImg[:,0]] = 255 - pointsImg[:,2]
    
        
    cv2.imwrite("Test_depth_img.png",depthimg)
    # Create a overlap image
    if (img.any()):
        #convert the image into HSV
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        # PLOT points of the image over the image
        for idx in range(1,pointsImg.shape[0]):
            img[pointsImg[idx,1],pointsImg[idx,0]] = (pointsImg[idx,2],255,255)
            cv2.circle(img, (pointsImg[idx,0],pointsImg[idx,1]),1, (int(pointsImg[idx,2]),255,255),-1)
        img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
        cv2.imwrite("TDepth_img_Overlap.png",img)
    
    return depthimg

def createTransformationMatrix(R,t):
    lastRow = np.array([0,0,0,1])
    t = np.reshape(t,(t.shape[0],1))
    rT = np.hstack((R,t))
    rT = np.vstack((rT,lastRow))
    return(rT)

def flipImgRowsNCols(imgCentroids):

    for idx in range(imgCentroids.shape[0]):
        temp = imgCentroids[idx][0]
        imgCentroids[idx][0] = imgCentroids[idx][1]
        imgCentroids[idx][1] = temp

    return(imgCentroids)

def extractCentroids(pointCloud):
        # Extract the centroids 
    centroids = np.empty((1,3))

    for idx in range(0,len(pointCloud)-1):
        for centroidno in range(0,pointCloud[idx][0].shape[0]):
            temp = pointCloud[idx][0][centroidno]
            array = np.array(temp).reshape(1,3)
            centroids = np.vstack((centroids,array))

    centroids = centroids[1:]
    return(centroids)


def refineTranformation(impPts, ptCldPts, initialRVec,  initialTVec, K, D):
    rVec, tVec = cv2.solvePnPRefineLM(ptCldPts, impPts, K, None, initialRVec, initialTVec)
    return (rVec,tVec)