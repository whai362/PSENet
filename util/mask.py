import cv2
import numpy as np

import util
from util import nb as neighbour

    
def find_white_components(mask, min_area = 0):
    mask = (mask == 0) * 1
    return find_black_components(mask, min_area);
    
def find_black_components(mask, min_area = 0):
    """
    find components of zeros. 
    mask is a 0-1 matrix, ndarray.
    """
    neighbour_type = neighbour.N4
    visited = mask.copy()
    c_mask = util.img.black(mask)

    root_idx = [1]
    def get_new_root():
        root_idx[0] += 1
        return root_idx[0]
        
    def is_visited(xy):
        x, y = xy
        return visited[y][x]
        
    def set_visited(xy):
        x, y = xy
        visited[y][x] = 255
    
    def set_root(xy, root):
        x, y = xy
        c_mask[y][x] = root
        
    def get_root(xy):
        x, y = xy
        return c_mask[y][x]
        
    rows, cols = np.shape(mask)
    q = []
    for y in xrange(rows):
        for x in xrange(cols):
            xy = (x, y)
            if  is_visited(xy):
                continue
                
            q.append(xy)
            new_root = get_new_root()
            while len(q) > 0:
                cp = q.pop()
                set_root(cp, new_root)
                set_visited(cp)
                nbs = neighbour.get_neighbours(cp[0], cp[1], cols, rows, neighbour_type)
                for nb in nbs:
                    if not is_visited(nb) and nb not in q:
#                         q.append(nb)
                        q.insert(0, nb)
    
    components = {}
    for y in xrange(rows):
        for x in xrange(cols):
            root = get_root((x, y))
            if root == 0:
                continue
                
            if root not in components:
                components[root] = []
                
            components[root].append((x,y))
    
    ret = []
    
    for root in components:
        if len(components[root]) >= min_area:
            ret.append(components[root])
        
    return ret
    

