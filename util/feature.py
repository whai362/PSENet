# encoding utf-8
def hog(img, bins =9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=False, feature_vector=True):
    """
    Extract hog feature from image.
    See detail at https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_hog.py
    """
    from skimage.feature import hog
    return hog(img, 
                orientations = bins, 
                pixels_per_cell = pixels_per_cell,
                cells_per_block = cells_per_block, 
                visualise = False, 
                transform_sqrt=False,
                feature_vector=True)
