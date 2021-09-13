import torch
import numpy as np
from torchvision import transforms

class dataLoader(torch.utils.data.Dataset):
    def __init__ (self, filename):
        self.datafile = filename
        file_descriptor = open(self.datafile,'r')
        #self.data = json.load(file_descriptor)
        self.data = file_descriptor.readlines()
        self.data = self.data[:100]


    def __len__(self):
        return(len(self.data))

    def __getitem__(self, key):
        return(self.getItem(key))

    def getItem(self, key):


        """
        """
        # Define preprocessing Pipeline
        imgTensorPreProc = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


        lineString = str(self.data[key]).split(' ')

        # Read from the file
        srcDepthImageFileName = lineString[0]
        targetDepthImageFileName = lineString[1] 
        srcIntensityImageFileName = lineString[2]
        targetIntensityImageFileName = lineString[3] 
        srcColorImageFileName = lineString[4]
        targetColorImageFileName = lineString[5]
        pointCldFileName = lineString[6]
        transform = np.array(lineString[7:]).astype(float).reshape(4,4)

        __, __, srcClrImg = readimgfromfilePIL(srcColorImageFileName)
        #srcClrImg = normalizePILImg(srcClrImg) # Bring it to the range of [0,1]
        srcClrImgNormalized = imgTensorPreProc(srcClrImg)

        return(srcClrImg,srcClrImgNormalized)
