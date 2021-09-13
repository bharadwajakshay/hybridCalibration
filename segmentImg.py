import torch
from utils import *

# class : Class No

Bicycle = 2
Car = 7
Motorcycle = 14
Person = 15

offset = 4

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()

def selectClass(image,category):
    empytImage = np.zeros(image.shape,dtype=np.uint8)
    carIdx = np.where(image==category) 
    carIMage = empytImage
    carIMage[carIdx] = 255

    
    # get connected points 
    retVal = cv2.connectedComponentsWithStats(carIMage)
    centroids = retVal[3].astype(np.int)
    label = retVal[1]
    stats = retVal[2]

    filtCentroids = []

    # filtering the centroids
    # if the starting point of the class == 0,0 consider it as background
    for idx in range(0,len(centroids)):
        # Check if this is a background pixel
        if not(stats[idx][0] == 0 and stats[idx][1] == 0):
            #make sure the area of the pixed is more than 1000 pixels
            if (stats[idx][-1]>256):
                filtCentroids.append(centroids[idx])

    return(label, filtCentroids)

def plotCentroids(image, centroids, category):

    # Convert the binary image into a BGR image
    #image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # select the color
    if category == Bicycle:
        color = (255,0,156)
        label = "Bicycle"
    elif category == Car:
        color = (0,0,255)
        label = "Car"
    elif category == Motorcycle:
        color = (124,152,8)
        label = "Motorcycle"
    else:
        color = (255,0,255)
        label = "Person"


    for idx in range(0,len(centroids)):
        centroid = (centroids[idx][0],centroids[idx][1])
        image = cv2.circle(image, centroid, 4, color,thickness=-4)
        image = cv2.putText(image, label, org=(centroids[idx][0]+offset,centroids[idx][1]+offset),fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=color)

    '''
    cv2.imshow('segments', image)
    cv2.waitKey(0)
    '''
    return(image)

def processImage(imageFilename, seqID, mode):

    # Get semantic segmentation network
    segImgNetwork = torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet101', pretrained=True)
    segImgNetwork.eval()
    # send the model to cuda
    segImgNetwork.to('cuda')

    __, __, srcClrImg = readimgfromfilePIL(imageFilename)

    imgTensorPreProc = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    normalizedImageTensor = imgTensorPreProc(srcClrImg)

    if mode:
        srcClrImg.save('.Debug/images/Original_image_'+str(seqID)+'.png',"PNG")
        
    normalizedImageTensor = normalizedImageTensor.unsqueeze(0)
    # send image to cuda
    normalizedImageTensor = normalizedImageTensor.to('cuda')
        
    with torch.no_grad():
        output = segImgNetwork(normalizedImageTensor)['out'][0]
        # output consists of both [out] and [aux]

        
    prediction = output.argmax(0)

        
    # Plot semantic segmentation
    # color the image
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plotting the semantic segmentation results
    plot = Image.fromarray(prediction.byte().cpu().numpy()).resize(srcClrImg.size)
    plot.putpalette(colors)
        
    plot.save('ImgSegmentation/SegmentedPointCloud.png')

    # write the output to file 
    prediction = prediction.byte().cpu().numpy()
    np.savetxt('prediction.txt',prediction)

    # Get centroid for the categories

    labelBicycle, centroidBicycle = selectClass(prediction, Bicycle)
    labelCar, centroidCar = selectClass(prediction, Car)
    labelMotorcycle, centroidMotorcycle = selectClass(prediction, Motorcycle)
    labelPerson, centroidPerson = selectClass(prediction, Person)

    # Plot the centroids
    semanticImage = cv2.imread('ImgSegmentation/SegmentedPointCloud.png')

    """
    cv2.imshow('Semantic Image',semanticImage)
    cv2.waitKey(0)
    """
    semanticImage = plotCentroids(semanticImage, centroidBicycle, Bicycle)
    semanticImage = plotCentroids(semanticImage, centroidCar, Car)
    semanticImage = plotCentroids(semanticImage, centroidMotorcycle, Motorcycle)
    semanticImage = plotCentroids(semanticImage, centroidPerson, Person)

    if mode:
        cv2.imwrite('.Debug/images/Semantic_Image_with_Centroid_'+str(seqID)+'.png',semanticImage)
 

    return(semanticImage, prediction, [[centroidBicycle, Bicycle],[centroidCar, Car],[centroidMotorcycle, Motorcycle],[centroidPerson, Person]],np.array(srcClrImg))
  
