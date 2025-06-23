#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

MODULE preprocessing

# This file contains a series of functions to process an array before extrac-
# ting faults. 
# This includes functions for: 
# (1) geotiff extraction
# (2) thresholding
# (3) skeletonize
# (4) labelling connected components
# (5) removing components
# (6) conversion to points. 

"""
# Packages
import cv2
import numpy as np
import rasterio
from skimage.morphology import skeletonize
# Custom package
#import cv_algorithms

import sys

# UNCOMMENT if you have downloaded directly the cv_algorithms library
# Replace your_directory by the directory where you downloaded the library
# cv_algo=("C://your_directory//cv_algorithms-master//cv_algorithms-master//cv_algorithms")
# sys.path.append(cv_algo)

from pathlib import Path
import os

# Fatbox
path_file=Path(__file__).absolute()
path_folder=path_file.parent
#print(path_folder)
os.chdir(path_folder) # make /modules as working directory

#Import Fatbox
import edits 
import metrics
import plots
import utils 
import structural_analysis


#******************************************************************************
# (1) Extract geotiff as array
#******************************************************************************

def extract_geotiff_and_coord (path_geotiff, coord_north=40000,coord_south=40100,coord_west=12000,coord_east=12150,system='grid coordinates'):
    """ Extract Geotiff as 3 np.array : data, latitudes, longitudes.
    
    (renew version of extract_dem_and_loc)

    This function extract the data layer from a  geotiff file.
    The information can be a Digital Elevation Model or the result of
    a hand mapping for example.
    It convert the .tiff file, which is georeferenced by latitude and 
    longitude, in numpy array.
    
    The geotiff are extracted as a rectangular grid, with 3 values per point
    ie. data, latitude and longitude given in 3 matrix : 
    data, latitudes, longitudes.
    
    The coordinates are the position of the corner of the rectangular
    choosen area. (Important to be rectangular !)
    To give the coordinates in latitude longitude, enter the exact coordinates
    which includes all the decimals number of the corner pixel.    

    Parameters
    ----------
    
    path_geotiff: str
        absolute or relative path of the geotiff.
        
    coord_north : the north coordinate
        FLOAT if a latitude
        INT if a pixel grid coordinate

    coord_south : the south coordinate
        FLOAT if a latitude
        INT if a pixel grid coordinate

    coord_west : the west coordinate
        FLOAT if longitude
        INT if a pixel grid coordinate

    coord_east : the east coordinate
        FLOAT if longitude
        INT if a pixel grid coordinate
        
    system : STR, optional, default is 'grid coordinates'
        Coordinate system in which the  corner position are given (ie system 
        of coord_north, coord_south, coord_west, coord_east). 
        Important  please give the coordinates all in the same system.
        Possible values:
        system = 'grid coordinates' if grid coordinates given in pixel.
        system= 'geographic' if longitude - latitude given.

    -------
    Returns
    -------
    np.array
    data,latitudes,longitudes

    """

    # Assertions
    assert os.path.isfile(path_geotiff)==True,'This file directory is incorrect'
    assert system in ['grid coordinates','geographic'], "'system' text incorrect" 
    
    raster=rasterio.open(path_geotiff)

    array_read=raster.read(1)

    # Select an interesting part, defined on base of observation on GIS software
    if system == 'geographic':
        coord_north,coord_west=raster.index(coord_west,coord_north)
        coord_south,coord_east=raster.index(coord_east,coord_south)
        coord_south=coord_south+1
        coord_east=coord_east+1

    data = array_read[coord_north:coord_south,coord_west:coord_east]

    # Create the longitude and latitude arrays.
    longitudes=np.zeros((coord_south-coord_north,coord_east-coord_west))
    latitudes=np.zeros((coord_south-coord_north,coord_east-coord_west))

    #icol is the matrix index of the column and irow index of row
    
    icol=0
    for col in range(coord_west,coord_east):
        irow=0
        for row in range(coord_north,coord_south):
            long,lat=raster.xy(row,col)
            longitudes[irow,icol]=long
            latitudes[irow,icol]=lat
            irow=irow+1
        icol=icol+1

    return data,latitudes,longitudes


#******************************************************************************
# (2) THRESHOLDING
# A couple of functions that allow you to threshold your data in different ways
#******************************************************************************

def simple_threshold_binary(arr, threshold):
    """ Thresholds array into a binary array
    
    Parameters
    ----------
    arr : np.array
        Input array that we binarize with threshold
    
    threshold : int, float
        The threshold used to binarize the input array
    
    Returns
    -------  
    arr
        Binarized output array (type: uint8)
    """
    
    # Assertions
    assert isinstance(arr, np.ndarray), "Input is not a NumPy array"
    assert isinstance(threshold, int) or isinstance(threshold, float), "Threshold is neither int nor float"
    
    # Calculation
    arr = np.where(arr > threshold, 1, 0)
    arr = np.uint8(arr)    
    
    return arr




def adaptive_threshold(arr):
    """ Thresholds array into a binary array using an adaptive threshold (Binary+Otsu)
    
    Parameters
    ----------
    
    arr : np.array
        Input array that we binarize with threshold
    
    Returns
    -------  
    arr
        Binarized output array (type: uint8)
    """    
    
    # Assertion
    assert isinstance(arr, np.ndarray), "Input is not a NumPy array"    

    # Calculation
    # Scale to [0,1]    
    arr = (arr-np.nanmin(arr))/(np.nanmax(arr)-np.nanmin(arr))
    # Scale to [0,255]
    arr = 255 * arr
    # Create image
    image = cv2.resize(arr.astype('uint8'), dsize=(arr.shape[1], arr.shape[0]))
    # Apply adaptive threshold
    _, arr = cv2.threshold(image,
                           0,
                           1,
                           cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Convert back to NumPy array
    arr = np.uint8(arr) 
    
    return arr




#******************************************************************************
# (3) SKELETONIZE
# A couple of functions that allow you to skeletonize your data, i.e. reduce to
# one pixel thick lines
#******************************************************************************

def skeleton_scipy(arr):
    """ Basic skeletonize function from SciPy
    
    Parameters
    ----------
    
    arr : np.array
        Input array
    
    Returns
    -------  
    arr
        Output array
    """
    
    # Assertion
    assert isinstance(arr, np.ndarray), "Input is not a NumPy array"    
    
    return skeletonize(arr)



def skeleton_guo_hall(arr):
    """ Optimized skeletonize function from cv_algorithms (https://github.com/ulikoehler/cv_algorithms)
    
    Parameters
    ----------
    
    arr : np.array
        Input array
    
    Returns
    -------  
    arr
        Output array
    """
    
    # Assertion
    assert isinstance(arr, np.ndarray), "Input is not a NumPy array" 
    
    # Calculation
    #arr = cv_algorithms.guo_hall(arr)
    arr = thinning.guo_hall(arr)
    
    # Correct edge effect
    arr[0, :] = arr[1, :]
    arr[-1,:] = arr[-2,:]
    arr[:, 0] = arr[:, 1]
    arr[:,-1] = arr[:,-2]
    
    return arr



#******************************************************************************
# (4) LABEL CONNECTED COMPONENTS
# A function to label connected components in array
#******************************************************************************

def connected_components(arr):
    """ Label connected components
    
    Parameters
    ----------
    
    arr : np.array
        Input array
    
    Returns
    -------  
    ret
        Output array
    markers
        Components
    """
    
    # Assertion
    assert isinstance(arr, np.ndarray), "Input is not a NumPy array"
    
    # Calculation
    ret, markers = cv2.connectedComponents(arr)
    
    return ret, markers



#******************************************************************************
# (5) REMOVAL
# A couple of functions to remove certain components
#******************************************************************************

def remove_small_regions(arr, size):
    """ Remove components below certain size
    
    Parameters
    ----------
    
    arr : np.array
        Input array
    size : int
    
    Returns
    -------  
    arr
        Output array
    """
    # Assertion
    assert isinstance(arr, np.ndarray), "Input array is not a NumPy array"
    assert isinstance(arr, int), "Input size is not an integer "
    
    
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(arr, connectivity=8)
    
    # connectedComponentswithStats yields every seperated component with
    # information on each of them, such as size
    # the following part is just taking out the background which is also
    # considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of
    # the sizes or whatever

    # your answer image
    arr = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= size:
            arr[output == i + 1] = 255

    # Convert to uint8
    arr = np.uint8(arr)    

    return arr




def remove_large_regions(arr, size):
    """ Remove components above certain size
    
    Parameters
    ----------
    
    arr : np.array
        Input array
    size : int
    
    Returns
    -------  
    arr
        Output array
    """
    # Assertion
    assert isinstance(arr, np.ndarray), "Input array is not a NumPy array"
    assert isinstance(arr, int), "Input size is not an integer "    
    
    
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(arr, connectivity=8)
    
    # connectedComponentswithStats yields every seperated component with
    # information on each of them, such as size
    # the following part is just taking out the background which is also
    # considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of
    # the sizes or whatever

    # your answer image
    arr = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] <= size:
            arr[output == i + 1] = 255

    # Convert to uint8
    arr = np.uint8(arr)    

    return arr




#******************************************************************************
# (6) CONVERSION
# A function to convert an array to points (x,y)
#******************************************************************************

def array_to_points(arr):
    """ A function to convert an array to points (x,y)
    
    Parameters
    ----------
    arr : np.array
        Input array that we binarize with threshold
    
    
    Returns
    -------  
    arr
        Output array (points)
    """
    
    # Assertions
    assert isinstance(arr, np.ndarray), "Input is not a NumPy array"
    
    # Calculation
    n = np.count_nonzero(arr)
    points = np.zeros((n, 2))
    (points[:, 1], points[:, 0]) = np.where(arr != 0)
    
    return points




