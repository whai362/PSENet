import log
import dtype
# import plt
import np
import img
_img = img
import dec
import rand
import mod
import proc
import test
import neighbour as nb
#import mask
import str_ as str
import io as sys_io
import io_ as io
import feature
import thread_ as thread
import caffe_ as caffe
# import tf
import cmd
import ml
import sys
import url
from .misc import *
from .logger import *
# log.init_logger('~/temp/log/log_' + get_date_str() + '.log')

def exit(code = 0):
    sys.exit(0)
    
is_main = mod.is_main
init_logger = log.init_logger

def sit(img, path = None, name = ""):
    if path is None:
        _count = get_count();
        path = '~/temp/no-use/images/%s_%d_%s.jpg'%(log.get_date_str(), _count, name)
      
    if type(img) == list:
        plt.show_images(images = img, path = path, show = False, axis_off = True, save = True)
    else:
        plt.imwrite(path, img)
    
    return path
_count = 0;

def get_count():
    global _count;
    _count += 1;
    return _count    

def cit(img, path = None, rgb = True, name = ""):
    _count = get_count();
    if path is None:
        img = np.np.asarray(img, dtype = np.np.uint8)
        path = '~/temp/no-use/%s_%d_%s.jpg'%(log.get_date_str(), _count, name)
        _img.imwrite(path, img, rgb = rgb)
    return path        

def argv(index):
    return sys.argv[index]
