import numpy as np
import os
import sys
import open3d as o3d
import yaml
from scipy import spatial
from utils import *


# class : Class No according to  image segmentation 

Bicycle = 2
Car = 7
Motorcycle = 14
Person = 15


def processPoint(filename, labelFile, colorMap, seqId, DEBUG):
    pointcloud = readvelodynepointcloud(filename)[:,:3]
    #npColorMap = np.zeros((pointcloud.shape[0],3))

    #display original Pointclouds
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud[:,:3])
    #pcd.colors = o3d.utility.Vector3dVector(npColorMap)
    if DEBUG:
        o3d.visualization.draw_geometries([pcd])

    label = np.fromfile(labelFile, dtype=np.int32).reshape((-1, 1))
    label = label & 0xFFFF  # delete high 16 digits binary

    pointcloud, label  = filterPointCloud(pointcloud, label)

    npColorMap = np.zeros((label.shape[0],3))

    for idx in range(0,label.shape[0]):
        category = label[idx][0]
        if(category == 252):
            npColorMap[idx] = (0,0,255) 
        else:
            color = colorMap.__getitem__(category)
            npColorMap[idx] = (color[2],color[1],color[0]) 


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(npColorMap)
    if DEBUG:
        o3d.visualization.draw_geometries([pcd])
        
    remIdx = np.empty(0)

    # find cars
    carIdx = np.where(label==10)
    cars = pointcloud[carIdx[0],:3]

    carIdx = np.where(label==252)
    cars = np.vstack((cars, pointcloud[carIdx[0],:3]))

    color = (0,0,255)
    colorCar = np.tile(color,(cars.shape[0],1))
    centroidsCar = getAllCentroids(cars)

    # Create car label vector
    carsLabelVec = np.ones((cars.shape[0],1)) * Car

    # Plot pooints 
    if DEBUG:
        plotPtCldNCentroids(cars, colorCar, centroidsCar,seqId)

   ############################################################

    # find Bicycle
    bicycleIdx = np.where(label ==11)
    bicycles = pointcloud[bicycleIdx[0],:3]

    bicycleIdx = np.where(label == 253)
    bicycles = np.vstack((bicycles,pointcloud[bicycleIdx[0],:3]))


    color = (0,255,0)
    colorBicycle = np.tile(color,(bicycles.shape[0],1))
    centroidBicycle = getAllCentroids(bicycles)

    # Create bicycles label vector
    bicyclesLabelVec = np.ones((bicycles.shape[0],1)) * Bicycle
    
    # Plot pooints 
    if DEBUG:
        plotPtCldNCentroids(bicycles, colorBicycle, centroidBicycle,seqId)

    ############################################################
        
    # find Motorcycle
    motorcycleIdx = np.where(label == 15)
    motorcycles = pointcloud[motorcycleIdx[0],:3]

    color = (51,0,51)
    colorMotorcycle = np.tile(color,(motorcycles.shape[0],1))
    centroidMotorcycle = getAllCentroids(motorcycles)

    # Create motorcycles label vector
    motorcyclesLabelVec = np.ones((motorcycles.shape[0],1)) * Motorcycle

    # Plot pooints
    if DEBUG:
        plotPtCldNCentroids(motorcycles, colorMotorcycle, centroidMotorcycle,seqId)

    ############################################################
        
    # find person
    personIdx = np.where(label == 30)
    persons = pointcloud[personIdx[0],:3]

    personIdx = np.where(label == 254)
    persons = np.vstack((persons,pointcloud[personIdx[0],:3]))

    color = (255,0,255)
    colorPersons = np.tile(color,(persons.shape[0],1))
    centroidPersons = getAllCentroids(persons)

    # Create persons label vector
    personsLabelVec = np.ones((persons.shape[0],1)) * Person

    # Plot pooints
    if DEBUG:
        plotPtCldNCentroids(persons, colorPersons, centroidPersons,seqId)

    ############################################################
    
    # find motorcyclist
    motorcyclistIdx = np.where(label == 32)
    motorcyclists = pointcloud[motorcyclistIdx[0],:3]

    motorcyclistIdx = np.where(label == 255)
    motorcyclists = np.vstack((motorcyclists,pointcloud[motorcyclistIdx[0],:3]))

    color = (255,0,255)
    colorMotorcyclists = np.tile(color,(motorcyclists.shape[0],1))
    centroidMotorcyclists = getAllCentroids(motorcyclists)

    # Create motorcyclists label vector
    motorcyclistsLabelVec = np.ones((motorcyclists.shape[0],1)) * Person

    # Plot pooints
    if DEBUG:
        plotPtCldNCentroids(motorcyclists, colorMotorcyclists, centroidMotorcyclists,seqId)

    ############################################################

    # find bicyclist    
    bicyclistIdx = np.where(label == 31)
    bicyclists = pointcloud[bicyclistIdx[0],:3]

    bicyclistIdx = np.where(label == 253)
    bicyclists = np.vstack((bicyclists,pointcloud[bicyclistIdx[0],:3]))

    color = (255,0,255)
    colorBicyclists = np.tile(color,(bicyclists.shape[0],1))
    centroidBicyclists = getAllCentroids(bicyclists)

    # Create bicyclists label vector
    bicyclistsLabelVec = np.ones((bicyclists.shape[0],1)) * Person

    # Plot pooints 
    if DEBUG:
        plotPtCldNCentroids(bicyclists, colorBicyclists, centroidBicyclists,seqId)

    ############################################################

    segmentedCloud = np.vstack((cars, bicycles, motorcycles, persons, motorcyclists, bicyclists))

    segmentedLabels = np.vstack((carsLabelVec, bicyclesLabelVec, 
                                    motorcyclesLabelVec, personsLabelVec,
                                    motorcyclistsLabelVec, bicyclistsLabelVec))

    segmentedCloudColor = np.vstack((colorCar, colorBicycle, colorMotorcycle, 
                                     colorPersons, colorMotorcyclists, colorBicyclists))



    if DEBUG:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(segmentedCloud)
        pcd.colors = o3d.utility.Vector3dVector(segmentedCloudColor)
        o3d.visualization.draw_geometries([pcd])
    

    # merge centroids from persons 
    centroidPerson = centroidPersons
    centroidPerson = np.append(centroidPerson,centroidBicyclists)
    centroidPerson = np.append(centroidPerson,centroidMotorcyclists)

    return(segmentedCloud, segmentedLabels, segmentedCloudColor, [[centroidBicycle, Bicycle],[centroidsCar, Car],[centroidMotorcycle, Motorcycle],[centroidPerson, Person]])