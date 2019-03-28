#encoding=utf-8

import numpy as np

N1 = 'n1'
N2 = 'n2'
N4 = 'n4'
N8 = 'n8'

def _in_image(c, w, h):
    cx, cy = c
    return cx >=0 and cx < w and cy >= 0 and cy < h

def n1(x, y, w, h):
    """down and right"""
    neighbours = []
    candidates = [(x, y + 1),  (x + 1, y)];
    
    for c in candidates:
        if _in_image(c, w, h):
            neighbours.append(c)
    
    return neighbours
    

def n2(x, y, w, h):
    neighbours = []
    candidates = [(x, y + 1),  (x + 1, y), (x + 1, y + 1), (x - 1, y + 1)];
    for c in candidates:
        if _in_image(c, w, h):
            neighbours.append(c)
    
    return neighbours;

def n4(x, y, w, h):
    neighbours = []
    candidates = [(x, y - 1),(x, y + 1),  (x + 1, y), (x - 1, y)];
    for c in candidates:
        if _in_image(c, w, h):
            neighbours.append(c)
    return neighbours
    

def n8(x, y, w, h):
    neighbours = []
    candidates = [(x + 1, y - 1),(x, y - 1),(x - 1, y - 1), (x - 1, y),(x, y + 1),  (x + 1, y), (x + 1, y + 1), (x - 1, y + 1)];
    for c in candidates:
        if _in_image(c, w, h):
            neighbours.append(c)
    
    return neighbours;
    
    
def n1_count(w, h):
    return 2 * w * h - w - h
    
def n2_count(w, h):
    return 4 * w * h - 3 * w - 3 * h + 2
    
    
_dict1 = {N1:n1, N2:n2, N4:n4, N8:n8};
_dict2 = {N1:n1_count, N2:n2_count};

def get_neighbours(x, y, w, h, neighbour_type):
    if neighbour_type in _dict1:
        fn = _dict1[neighbour_type]
        return fn(x, y, w, h)
    raise NotImplementedError("unknown neighbour type '%s'" % (neighbour_type))
    
def count_neighbours(w, h, neighbour_type):
    if neighbour_type in _dict2:
        fn = _dict2[neighbour_type]
        return fn(w, h)
    raise NotImplementedError("unknown neighbour type '%s'" % (neighbour_type))
    

if __name__ == "__main__":
    w, h = 10, 10
    np.testing.assert_equal(len(n4(0, 0, w, h)), 2)
    np.testing.assert_equal(len(n8(0, 0, w, h)), 3)
    
    np.testing.assert_equal(len(n4(0, 2, w, h)), 3)
    np.testing.assert_equal(len(n8(0, 2, w, h)), 5)
    
    np.testing.assert_equal(len(n4(3, 3, w, h)), 4)
    np.testing.assert_equal(len(n8(3, 3, w, h)), 8)
