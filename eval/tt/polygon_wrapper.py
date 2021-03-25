
import numpy as np
from skimage.draw import polygon

"""
:param det_x: [1, N] Xs of detection's vertices
:param det_y: [1, N] Ys of detection's vertices
:param gt_x: [1, N] Xs of groundtruth's vertices
:param gt_y: [1, N] Ys of groundtruth's vertices
##############
All the calculation of 'AREA' in this script is handled by:
1) First generating a binary mask with the polygon area filled up with 1's
2) Summing up all the 1's
"""


def area(x, y):
    """
    This helper calculates the area given x and y vertices.
    """
    ymax = np.max(y)
    xmax = np.max(x)
    bin_mask = np.zeros((ymax, xmax))
    rr, cc = polygon(y, x)
    bin_mask[rr, cc] = 1
    area = np.sum(bin_mask)
    return area
    #return np.round(area, 2)


def approx_area_of_intersection(det_x, det_y, gt_x, gt_y):
    """
    This helper determine if both polygons are intersecting with each others with an approximation method.
    Area of intersection represented by the minimum bounding rectangular [xmin, ymin, xmax, ymax]
    """
    det_ymax = np.max(det_y)
    det_xmax = np.max(det_x)
    det_ymin = np.min(det_y)
    det_xmin = np.min(det_x)

    gt_ymax = np.max(gt_y)
    gt_xmax = np.max(gt_x)
    gt_ymin = np.min(gt_y)
    gt_xmin = np.min(gt_x)

    all_min_ymax = np.minimum(det_ymax, gt_ymax)
    all_max_ymin = np.maximum(det_ymin, gt_ymin)

    intersect_heights = np.maximum(0.0, (all_min_ymax - all_max_ymin))

    all_min_xmax = np.minimum(det_xmax, gt_xmax)
    all_max_xmin = np.maximum(det_xmin, gt_xmin)
    intersect_widths = np.maximum(0.0, (all_min_xmax - all_max_xmin))

    return intersect_heights * intersect_widths

def area_of_intersection(det_x, det_y, gt_x, gt_y):
    """
    This helper calculates the area of intersection.
    """
    if approx_area_of_intersection(det_x, det_y, gt_x, gt_y) > 1: #only proceed if it passes the approximation test
        ymax = np.maximum(np.max(det_y), np.max(gt_y)) + 1
        xmax = np.maximum(np.max(det_x), np.max(gt_x)) + 1
        bin_mask = np.zeros((ymax, xmax))
        det_bin_mask = np.zeros_like(bin_mask)
        gt_bin_mask = np.zeros_like(bin_mask)

        rr, cc = polygon(det_y, det_x)
        det_bin_mask[rr, cc] = 1

        rr, cc = polygon(gt_y, gt_x)
        gt_bin_mask[rr, cc] = 1

        final_bin_mask = det_bin_mask + gt_bin_mask

        inter_map = np.where(final_bin_mask == 2, 1, 0)
        inter = np.sum(inter_map)
        return inter
#        return np.round(inter, 2)
    else:
        return 0


def iou(det_x, det_y, gt_x, gt_y):
    """
    This helper determine the intersection over union of two polygons.
    """

    if approx_area_of_intersection(det_x, det_y, gt_x, gt_y) > 1: #only proceed if it passes the approximation test
        ymax = np.maximum(np.max(det_y), np.max(gt_y)) + 1
        xmax = np.maximum(np.max(det_x), np.max(gt_x)) + 1
        bin_mask = np.zeros((ymax, xmax))
        det_bin_mask = np.zeros_like(bin_mask)
        gt_bin_mask = np.zeros_like(bin_mask)

        rr, cc = polygon(det_y, det_x)
        det_bin_mask[rr, cc] = 1

        rr, cc = polygon(gt_y, gt_x)
        gt_bin_mask[rr, cc] = 1

        final_bin_mask = det_bin_mask + gt_bin_mask

        #inter_map = np.zeros_like(final_bin_mask)
        inter_map = np.where(final_bin_mask == 2, 1, 0)
        inter = np.sum(inter_map)

        #union_map = np.zeros_like(final_bin_mask)
        union_map = np.where(final_bin_mask > 0, 1, 0)
        union = np.sum(union_map)
        return inter / float(union + 1.0)
        #return np.round(inter / float(union + 1.0), 2)
    else:
        return 0

def iod(det_x, det_y, gt_x, gt_y):
    """
    This helper determine the fraction of intersection area over detection area
    """

    if approx_area_of_intersection(det_x, det_y, gt_x, gt_y) > 1: #only proceed if it passes the approximation test
        ymax = np.maximum(np.max(det_y), np.max(gt_y)) + 1
        xmax = np.maximum(np.max(det_x), np.max(gt_x)) + 1
        bin_mask = np.zeros((ymax, xmax))
        det_bin_mask = np.zeros_like(bin_mask)
        gt_bin_mask = np.zeros_like(bin_mask)

        rr, cc = polygon(det_y, det_x)
        det_bin_mask[rr, cc] = 1

        rr, cc = polygon(gt_y, gt_x)
        gt_bin_mask[rr, cc] = 1

        final_bin_mask = det_bin_mask + gt_bin_mask

        inter_map = np.where(final_bin_mask == 2, 1, 0)
        inter = np.round(np.sum(inter_map), 2)

        det = np.round(np.sum(det_bin_mask), 2)
        return inter / float(det + 1.0)
        #return np.round(inter / float(det + 1.0), 2)
    else:
        return 0
