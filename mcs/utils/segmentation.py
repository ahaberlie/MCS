import uuid
import os

import numpy as np

from scipy import ndimage, sparse

from scipy.ndimage import binary_closing, binary_dilation
from scipy.ndimage.measurements import label

from skimage.morphology import disk, watershed, remove_small_objects
from skimage.measure import regionprops

from skimage.segmentation import find_boundaries
from scipy.ndimage.morphology import binary_fill_holes



def get_qualifying_clusters(rImage, strat_dbz, conv_dbz, int_dbz, min_length, 
                            conv_buffer, min_size=10, strat_buffer=0):
    """Combines the logic of get_intense_cells, 
    connect_intense_cells, and connect_stratiform_to_lines
    to return pixels associated with qualifying slices.
    
    Stratiform >= 4 (20 dBZ)
    Convection >= 8 (40 dBZ)
    Intense >= 10 (50 dBZ)
    
    Parameters
    ----------
    rImage: (N, M) ndarray
        Radar Image from which to extract qualifying lines.
        
    strat_dbz: int
        Threshold used to identify stratiform pixels 
        (Multiply value by 5 to get dBZ)
       
    conv_dbz: int
        Threshold used to identify convective pixels 
        (Multiply value by 5 to get dBZ)
        
    int_dbz: int
        Threshold used to identify intense pixels
        (Multiply value by 5 to get dBZ)
        
    min_length: int
        Minimum length for a qualifying merged lines
        (Multiply value by 2 to get km)
        
    conv_buffer: int
        Distance within which intense cells are merged
        (Multiply value by 2 (pixel distance to km) and then
         multiply by minimum search disk radius (3) to get
         buffer size in km)
    
    min_size: int
        Minimum size for an intense cell to be considered in
        line-building process.
        
    strat_buffer: int
        Distance within which stratiform pixels are merged
        with qualifying merged lines.
        (Multiply value by 2 to account for pixel distance
        and then multiply by minimum search disk radius of 3
        to get buffer size in km)
         
    conv_buffer: integer
        Distance to search for nearby intense cells.

    Returns
    -------
    regions: list
        A list of regionprops for each qualifying slice.
        See scikit-image.measure.regionprops for more information.
    """

    convection = 1 * (rImage >= conv_dbz)

    stratiform = 1 * (rImage >= strat_dbz)

    labeled_image, _ = label(convection, np.ones((3,3), dtype=int))

    remove_small_objects(labeled_image, min_size=min_size, connectivity=2, in_place=True)
    
    regions = regionprops(labeled_image, intensity_image=rImage)

    for region in regions:
        if np.max(region.intensity_image) < int_dbz:
        
            ymin, xmin = np.min(region.coords[:, 0]), np.min(region.coords[:, 1])
            y, x = np.where(region.intensity_image > 0)
            labeled_image[ymin+y, xmin+x] = 0

    thresholded_image = 1 * binary_closing(labeled_image > 0, structure=disk(3), iterations=int(conv_buffer))

    labeled_image, _ = label(thresholded_image, np.ones((3,3)))

    regions = regionprops(labeled_image, intensity_image=rImage)

    for region in regions:
        if region.major_axis_length < min_length:
        
            ymin, xmin = np.min(region.coords[:, 0]), np.min(region.coords[:, 1])
            y, x = np.where(region.intensity_image > 0)
            labeled_image[ymin+y, xmin+x] = 0
    
    strat_mask = 1 * stratiform * (binary_dilation(1*(labeled_image > 0), structure=disk(3), iterations=strat_buffer))
    
    thresholded_image = 1*(labeled_image>0) + strat_mask
    
    #thresholded_image = watershed(strat_mask, labeled_image, mask=strat_mask)
    
    labeled_image, _ = label(1*(thresholded_image > 0), np.ones((3,3)))
    
    labeled_image *= stratiform
    
    regions = regionprops(labeled_image, intensity_image=thresholded_image)
    
    for region in regions:
        if np.max(region.intensity_image) < 2:
        
            ymin, xmin = np.min(region.coords[:, 0]), np.min(region.coords[:, 1])
            y, x = np.where(region.intensity_image > 0)
            labeled_image[ymin+y, xmin+x] = 0

    return regionprops(labeled_image, intensity_image=rImage)

    
def find_lines(rImage, conv_buffer, min_length=50):
    """Combines the logic of get_intense_cells and 
    connect_intense_cells to return pixels associated
    with qualifying merged lines.
    
    Stratiform >= 4 (20 dBZ)
    Convection >= 8 (40 dBZ)
    Intense >= 10 (50 dBZ)
    
    Parameters
    ----------
    rImage: (N, M) ndarray
        Radar Image from which to extract qualifying lines.
        
    conv_buffer: integer
        Distance to search for nearby intense cells.
        
    min_length: integer
        Minimum size requirment to be considered an MCS.
        Default is 50 (100 km with 2 km pixels)

    Returns
    -------
    labeled_image: (N, M) ndarray
        Binary image of pixels in qualifying merged lines. 
        Same dimensions as rImage.
    """
    
    convection = 1 * (rImage >= 8)

    stratiform = 1 * (rImage >= 4)

    labeled_image, _ = label(convection, np.ones((3,3), dtype=int))

    remove_small_objects(labeled_image, min_size=10, connectivity=2, in_place=True)
    
    regions = regionprops(labeled_image, intensity_image=rImage)

    for region in regions:
        if np.max(region.intensity_image) < 10:
        
            ymin, xmin = np.min(region.coords[:, 0]), np.min(region.coords[:, 1])
            y, x = np.where(region.intensity_image > 0)
            labeled_image[ymin+y, xmin+x] = 0

    thresholded_image = 1 * binary_closing(labeled_image > 0, structure=disk(3), iterations=int(conv_buffer))

    labeled_image, _ = label(thresholded_image, np.ones((3,3)))

    regions = regionprops(labeled_image, intensity_image=rImage)

    for region in regions:
        if region.major_axis_length < min_length:
        
            ymin, xmin = np.min(region.coords[:, 0]), np.min(region.coords[:, 1])
            y, x = np.where(region.intensity_image > 0)
            labeled_image[ymin+y, xmin+x] = 0
    
    return labeled_image


def get_intense_cells(rImage, min_size=10):
    """Return pixel coordinates and unique labels associated 
    with intense thunderstorm cells.
    
    Convection >= 8 (40 dBZ)
    Intense >= 10 (50 dBZ)
    
    Parameters
    ----------
    rImage: (N, M) ndarray
        Radar Image from which to extract intense cells.

    Returns
    -------
    labeled_image1: (N, M) ndarray
        Labeled image of intense cells. Same dimensions as rImage.
    """
    
    convection = np.uint8(rImage >= 8)

    labeled_image, _ = label(convection, np.ones((3,3)))

    remove_small_objects(labeled_image, min_size=min_size, connectivity=2, in_place=True)
    
    regions = regionprops(labeled_image, intensity_image=rImage)

    labeled_image1 = np.zeros(labeled_image.shape, dtype=int)

    for region in regions:
        if np.max(region.intensity_image) >= 10:
            labeled_image1 += (labeled_image == region.label) * rImage
            
    return labeled_image1
    

def connect_intense_cells(int_cells, conv_buffer):
    """Merge nearby intense cells if they are within a given
    convective region search radius.
    
    Parameters
    ----------
    int_cells: (N, M) ndarray
        Pixels associated with intense cells.
        
    conv_buffer: integer
        Distance to search for nearby intense cells.

    Returns
    -------
    labeled_image1: (N, M) ndarray
        Binary image of merged intense cells. Same dimensions as int_cells.
    """
    
    return binary_closing(int_cells>0, structure=disk(3), iterations=conv_buffer)


def connect_stratiform_to_lines(lines, stratiform, strat_buffer):
    """Connect pixels with values of 20 dBZ or greater surrounding 
    merged lines within a given stratiform search radius.
    
    Parameters
    ----------
    lines: (N, M) ndarray
        Pixels associated with merged lines.
        
    stratiform: (N, M) ndarray
        Binary image using a threshold of 20 dBZ.
        
    strat_buffer: integer
        Distance to search for stratiform pixels to 
        connect to merged lines.

    Returns
    -------
    labeled_image: (N, M) ndarray
        Labeled image where each slice has a unique value. 
        Has same dimensions as lines and stratiform.
    """
    
    strat_mask = 1 * stratiform * (binary_dilation(1*(lines > 0), structure=disk(3), iterations=strat_buffer))
    
    thresholded_image = 1*(lines>0) + strat_mask
    
    labeled_image, _ = label(1*(thresholded_image > 0), np.ones((3,3)))
    
    labeled_image *= stratiform
    
    regions = regionprops(labeled_image, intensity_image=thresholded_image)
    
    for region in regions:
        if np.max(region.intensity_image) < 2:
        
            ymin, xmin = np.min(region.coords[:, 0]), np.min(region.coords[:, 1])
            y, x = np.where(region.intensity_image > 0)
            labeled_image[ymin+y, xmin+x] = 0

    return labeled_image
