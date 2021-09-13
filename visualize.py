import numpy as np
import cv2

# display points on image

def displayPointsOnImage(ptCld, color, origImage,filename,centroid=None,segment=True):

    """
    # ptCLD = Nx6 where 0:3 = xyz
                    and  3:6 = RGB
    # Image = HXWx3 where 3 = RGB
    # P = 4x3 => [R|t]
    """

    '''
    Step1: Get the 2D image pts from 3D points
    Step2: Filter 3D and 3D points
    Step3: Show the points on the image
    '''

    # PLOT points of the image over the image

    image = np.copy(origImage)


 
    for idx in range(1,ptCld.shape[0]):
        image[ptCld[idx,1],ptCld[idx,0]] = (color[idx][2],color[idx][1],color[idx][0])
        image = cv2.circle(image,(ptCld[idx,0],ptCld[idx,1]),radius=1,color=(255,0,0),thickness=-1)

    if centroid is not None:
        for idx in range(0,centroid.shape[0]):
            image = cv2.circle(image,(centroid[idx,0],centroid[idx,1]),radius=5,color=(255,255,255),thickness=-1)
    
    if not segment:
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

    cv2.imwrite(filename,image)
