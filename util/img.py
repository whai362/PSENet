#coding=utf-8
'''
@author: dengdan
'''
import cv2
import numpy as np
import logging
import math
import event
import util

IMREAD_GRAY = 0
IMREAD_COLOR = 1
IMREAD_UNCHANGED = -1



COLOR_WHITE =(255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GREEN = (0, 255, 0)

COLOR_RGB_RED = (255, 0, 0)
COLOR_BGR_RED = (0, 0, 255)

COLOR_RGB_BLUE = (0, 0, 255)
COLOR_BGR_BLUE = (255, 0, 0)

COLOR_RGB_YELLOW = (255, 255, 0)
COLOR_BGR_YELLOW = (0, 255, 255)


COLOR_RGB_GRAY = (47, 79, 79)

COLOR_RGB_PINK = (255, 192, 203)
def imread(path, rgb = False, mode = cv2.IMREAD_COLOR):
    path = util.io.get_absolute_path(path)
    img = cv2.imread(path, mode)
    if img is None:
        raise IOError('File not found:%s'%(path))
        
    if rgb:
        img = bgr2rgb(img)
    return img

def imshow(winname, img, block = True, position = None, maximized = False, rgb = False):
    if isinstance(img, str):
        img = imread(path = img)
    
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    if rgb:
        img = rgb2bgr(img)
    cv2.imshow(winname, img)
    if position is not None:
#         cv2.moveWindow(winname, position[0], position[1])
        move_win(winname, position)
    
    if maximized:
        maximize_win(winname)  
        
        
    if block:
#         cv2.waitKey(0)
        event.wait_key(" ")
        cv2.destroyAllWindows()


def imwrite(path, img, rgb = False):
    if rgb:
        img = rgb2bgr(img)
    path = util.io.get_absolute_path(path)
    util.io.make_parent_dir(path)
    cv2.imwrite(path, img)

def move_win(winname, position = (0, 0)):
    """
    move pyplot window
    """
    cv2.moveWindow(winname, position[0], position[1])

def maximize_win(winname):
    cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, True);

def eq_color(target, color):
    for i, c in enumerate(color):
        if target[i] != color[i]:
            return False
    return True
    
def is_white(color):
    for c in color:
        if c < 255:
            return False
    return True
    
def black(shape):
    if len(np.shape(shape)) >= 2:
        shape = get_shape(shape)
    shape = [int(v) for v in shape]
    return np.zeros(shape, np.uint8)
    
def white(shape, value = 255):
    if len(np.shape(shape)) >= 2:
        shape = get_shape(shape)
    return np.ones(shape, np.uint8) * np.uint8(value)
    
def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def bgr2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def ds_size(image_size, kernel_size, stride):
    """calculate the size of downsampling result"""
    image_x, image_y = image_size
    

    kernel_x, kernel_y = kernel_size
    stride_x, stride_y = stride
    
    def f(iw, kw, sw):
        return int(np.floor((iw - kw) / sw) + 1)
    
    output_size = (f(image_x, kernel_x, stride_x), f(image_y, kernel_y, stride_y))
    return output_size


    
def get_roi(img, p1, p2):
    """
    extract region of interest from an image.
    p1, p2: two tuples standing for two opposite corners of the rectangle bounding the roi. 
    Their order is arbitrary.
    """
    x1, y1 = p1
    x2, y2 = p2
    
    x_min = min([x1, x2])
    y_min = min([y1, y2])
    x_max = max([x1, x2]) + 1
    y_max = max([y1, y2]) + 1
    
    return img[y_min: y_max, x_min: x_max]
    
def rectangle(img, left_up, right_bottom, color, border_width = 1):
    left_up = (int(left_up[0]), int(left_up[1]))
    right_bottom = (int(right_bottom[0]), int(right_bottom[1]))
    cv2.rectangle(img, left_up, right_bottom, color, border_width)


def circle(img, center, r, color, border_width = 1):
    center = (int(center[0]), int(center[1]))
    cv2.circle(img, center, r, color, border_width)

def render_points(img, points, color):
    for p in points:
        x, y = p
        img[y][x] = color

    
def draw_contours(img, contours, idx = -1, color = 1, border_width = 1):
#     img = img.copy()
    cv2.drawContours(img, contours, idx, color, border_width)
    return img

def get_contour_rect_box(contour):
    x,y,w,h = cv2.boundingRect(contour)
    return x, y, w, h

def get_contour_region_in_rect(img, contour):
    x, y, w, h = get_contour_rect_box(contour)
    lu, rb = (x, y), (x + w, y + h)
    return get_roi(img, lu, rb)

def get_contour_min_area_box(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    return box

def get_contour_region_in_min_area_rect(img, cnt):
    # find the min area rect of contour
    rect = cv2.minAreaRect(cnt)
    angle = rect[-1]
    box = cv2.cv.BoxPoints(rect)
    box_cnt = points_to_contour(box)
    
    # find the rectangle containing box_cnt, and set it as ROI
    outer_rect = get_contour_rect_box(box_cnt)
    x, y, w, h = outer_rect
    img = get_roi(img, (x, y), (x + w,  y + h))
    box = [(ox - x, oy - y) for (ox, oy) in box]
    
    # rotate ROI and corner points
    rows, cols = get_shape(img)
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, scale = 1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    bar_xy = np.hstack((box, np.ones((4, 1))))
    new_corners = np.dot(M, np.transpose(bar_xy))
    new_corners = util.dtype.int(np.transpose(new_corners))
#     cnt = points_to_contour(new_corners)
    
    xs = new_corners[:, 0]
    ys = new_corners[:, 1]
    lu = (min(xs), min(ys))
    rb = (max(xs), max(ys))
    return get_roi(dst, lu, rb)


def contour_to_points(contour):
    return np.asarray([c[0] for c in contour])


def points_to_contour(points):
    contours = [[list(p)]for p in points]
    return np.asarray(contours, dtype = np.int32)

def points_to_contours(points):
    return np.asarray([points_to_contour(points)])

def get_contour_region_iou(I, cnt1, cnt2):
    """
    calculate the iou of two contours
    """
    mask1 = util.img.black(I)
    draw_contours(mask1, [cnt1], color = 1, border_width = -1)
    
    mask2 = util.img.black(I)
    draw_contours(mask2, [cnt2], color = 1, border_width = -1)
    
    union_mask = ((mask1 + mask2) >=1) * 1
    intersect_mask = (mask1 * mask2 >= 1) * 1
    
    return np.sum(intersect_mask) * 1.0 / np.sum(union_mask)

    
def fill_bbox(img, box, color = 1):
    """
    filling a bounding box with color.
    box: a list of 4 points, in clockwise order, as the four vertice of a bounding box
    """
    util.test.assert_equal(np.shape(box), (4, 2))
    cnt = to_contours(box)
    draw_contours(img, cnt, color = color, border_width = -1)
    
def get_rect_points(left_up, right_bottom):
    """
    given the left up and right bottom points of a rectangle, return its four points
    """
    right_bottom, left_up = np.asarray(right_bottom), np.asarray(left_up)
    w, h = right_bottom - left_up
    x, y = left_up
    points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return points
    
def rect_perimeter(left_up, right_bottom):
    """
    calculate the perimeter of the rectangle described by its left-up and right-bottom point.
    """
    return sum(np.asarray(right_bottom) -  np.asarray(left_up)) * 2

def rect_area(left_up, right_bottom):
    wh = np.asarray(right_bottom) - np.asarray(left_up) + 1
    return np.prod(wh)
    
def apply_mask(img, mask):
    """
    the img will be masked in place. 
    """
    c = np.shape(img)[-1]
    for i in range(c):
        img[:, :, i] = img[:, :, i] * mask 
    return img
    
def get_shape(img):
    """
    return the height and width of an image
    """
    return np.shape(img)[0:2]

def get_wh(img):
    return np.shape(img)[0:2][::-1]

def get_value(img, x, y = None):
    if y == None:
        y = x[1]
        x = x[0]
    
    return img[y][x]        
    
def set_value(img, xy, val):
    x, y = xy
    img[y][x] = val


def filter2D(img, kernel):
    dst = cv2.filter2D(img, -1, kernel)
    return dst

def average_blur(img, shape = (5, 5)):
    return cv2.blur(img, shape)

def gaussian_blur(img, shape = (5, 5), sigma = 0):
    # sigma --> sigmaX, sigmaY
    blur = cv2.GaussianBlur(img,shape, sigma)
    return blur

def bilateral_blur(img, d = 9, sigmaColor = 75, sigmaSpace = 75):
    dst = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
    return dst

BLUR_AVERAGE =  'average'
BLUR_GAUSSIAN = 'gaussian'
BLUR_BILATERAL = 'bilateral'


_blur_dict = {
              BLUR_AVERAGE: average_blur,
              BLUR_GAUSSIAN: gaussian_blur,
              BLUR_BILATERAL: bilateral_blur
}

def blur(img, blur_type):
    fn = _blur_dict[blur_type]
    return fn(img)
    
def put_text(img, text, pos, scale = 1, color = COLOR_WHITE, thickness = 1):
    pos = np.int32(pos)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img = img, text = text, org = tuple(pos), fontFace = font,  fontScale = scale,  color = color, thickness = thickness)

def resize(img, f = None, fx = None, fy = None, size = None, interpolation = cv2.INTER_LINEAR):
    """
    size: (w, h)
    """
    h, w = get_shape(img)
    if fx != None and fy != None:
        return cv2.resize(img, None, fx = fx, fy = fy, interpolation = interpolation)
        
    if size != None:
        size = util.dtype.int(size)
#         size = (size[1], size[0])
        size = tuple(size)
        return cv2.resize(img, size, interpolation = interpolation)
    
    return cv2.resize(img, None, fx = f, fy = f, interpolation = interpolation)

def translate(img, delta_x, delta_y, size = None):
    M = np.float32([[1,0, delta_x],[0,1, delta_y]])
    if size == None:
        size = get_wh(img)
    
    dst = cv2.warpAffine(img,M, size)
    return dst


def rotate_about_center(src, angle, scale=1.):
    """https://www.oschina.net/translate/opencv-rotation"""
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4), rot_mat


def get_rect_iou(rects1, rects2):
    """
    calculate the iou between rects1 and rects2
    each rect consists of four points:[min_x, min_y, max_x, max_y]
    return: a iou matrix, len(rects1) * len(rects2)
    """
    rects1, rects2 = np.asarray(rects1), np.asarray(rects2)

    def _to_matrix(p, ps):
        p = np.ones((len(ps), 1)) * p
        ps = np.reshape(ps, (len(ps), 1))
        temp =np.hstack([p, ps])
        return temp
    
    def _get_max(p, ps):
        return np.max(_to_matrix(p, ps), axis = 1)
    
    def _get_min(p, ps):
        return np.min(_to_matrix(p, ps), axis = 1)
 
    
    def _get_area(rect):
        w, h = rect[:, 2] - rect[:, 0] + 1.0 , rect[:, 3] - rect[:, 1] + 1.0
        return w * h

    def _get_inter(rect1, rects2):
            x1 = _get_max(rect1[0], rects2[:, 0])
            y1 = _get_max(rect1[1], rects2[:, 1])
            
            x2 = _get_min(rect1[2], rects2[:, 2])
            y2 = _get_min(rect1[3], rects2[:, 3])
        
            w,h = x2-x1 +1, y2 - y1 + 1
            areas = w * h
            areas[np.where(w < 0)] = 0
            areas[np.where(h < 0)] = 0
            return areas
            
    area2 = _get_area(rects2)
    area1 = _get_area(rects1)
    iou = np.zeros((len(rects1), len(rects2)))
    for ri in range(len(rects1)):
        inter = _get_inter(rects1[ri, :], rects2)
        union = area1[ri] + area2 - inter
        iou[ri, :] = np.transpose( inter / union)
    return iou

def find_contours(mask):
    mask = np.asarray(mask, dtype = np.uint8)
    mask = mask.copy()
    contours, _ = cv2.findContours(mask, mode = cv2.RETR_CCOMP, 
                                   method = cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_two_level_contours(mask):
    mask = mask.copy()
    contours, tree = cv2.findContours(mask, mode = cv2.RETR_CCOMP, 
                                  method = cv2.CHAIN_APPROX_SIMPLE)
    return contours, tree
    
    
def is_in_contour(point, cnt):
    """tell whether a point is in contour or not. 
            In-contour here includes both the 'in contour' and 'on contour' cases.
       point:(x, y)
       cnt: a cv2 contour
    """
    # doc of pointPolygonTest: http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=pointpolygontest#cv.PointPolygonTest
    # the last argument means only tell if in or not, without calculating the shortest distance
    in_cnt = cv2.pointPolygonTest(cnt, point, False)
    return in_cnt >= 0;

def convex_hull(contour):
    hull = cv2.convexHull(contour, returnPoints=1)
    return hull
    
def random_color_3():
    c = util.rand.randint(low = 0, high = 255, shape = (3, ))
#     c = np.uint8(c)
    return c

def get_contour_area(cnt):
    return cv2.contourArea(cnt)
    
def is_valid_jpg(jpg_file):
    with open(jpg_file, 'rb') as f:     
        f.seek(-2, 2)
        return f.read() == '\xff\xd9'



def rotate_point_by_90(x, y, k, w = 1.0, h = 1.0):
    """
    Rotate a point xy on an image by k * 90
    degrees.
    Params:
        x, y: a point, (x, y). If not normalized within 0 and 1, the 
            width and height of the image should be specified clearly.
        w, h: the width and height of image
        k: k * 90 degrees will be rotated
    """
    k = k % 4
    
    if k == 0:
        return x, y
    elif k == 1:
        return y, w - x
    elif k == 2:
        return w - x, h - y
    elif k == 3:
        return h - y, x
    
    
def min_area_rect(xs, ys):
    """
    Args:
        xs: numpy ndarray with shape=(N,4). N is the number of oriented bboxes. 4 contains [x1, x2, x3, x4]
        ys: numpy ndarray with shape=(N,4), [y1, y2, y3, y4]
            Note that [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] can represent an oriented bbox.
    Return:
        the oriented rects sorrounding the box, in the format:[cx, cy, w, h, theta]. 
    """
    xs = np.asarray(xs, dtype = np.float32)
    ys = np.asarray(ys, dtype = np.float32)
        
    num_rects = xs.shape[0]
    box = np.empty((num_rects, 5))#cx, cy, w, h, theta
    for idx in xrange(num_rects):
        points = zip(xs[idx, :], ys[idx, :])
        cnt = points_to_contour(points)
        rect = cv2.minAreaRect(cnt)
        cx, cy = rect[0]
        w, h = rect[1]
        theta = rect[2]
        box[idx, :] = [cx, cy, w, h, theta]
    
    box = np.asarray(box, dtype = xs.dtype)
    return box
