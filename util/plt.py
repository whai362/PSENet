#coding=utf-8
'''
Created on 2016-9-27

@author: dengdan
'''
import matplotlib.pyplot as plt
import numpy as np
import util
        
def hist(x, title = None, normed = False, show = True, save = False, save_path = None, bin_count = 100, bins = None):    
    x = np.asarray(x)
    if len(np.shape(x)) > 1:
#         x = np.reshape(x, np.prod(x.shape))
        x = util.np.flatten(x)
    if bins == None:
        bins = np.linspace(start = min(x), stop = max(x), num = bin_count, endpoint = True, retstep = False)
    plt.figure(num = title)
    plt.hist(x, bins, normed = normed)
    if save:
        if save_path is None:
            raise ValueError
        path = util.io.join_path(save_path, title + '.png')
        save_image(path)
    if show:
        plt.show()
        #util.img.imshow(title, path, block = block)        

def plot_solver_data(solver_path):
    data = util.io.load(solver_path)
    training_losses = data.training_losses
    training_accuracies = data.training_accuracies
    val_losses = data.val_losses
    val_accuracies = data.val_accuracies
    plt.figure(solver_path)
    
    n = len(training_losses)
    x = range(n)
    
    plt.plot(x, training_losses, 'r-', label = 'Training Loss')
    
    if len(training_accuracies) > 0:
        plt.plot(x, training_accuracies, 'r--', label = 'Training Accuracy')
    
    if len(val_losses) > 0:
        n = len(val_losses)
        x = range(n)
        plt.plot(x, val_losses, 'g-', label = 'Validation Loss')
        
        if len(val_accuracies) > 0:
            plt.plot(x, val_accuracies, 'g--', label = 'Validation Accuracy')
    plt.legend()
    plt.show()
    
    
def rectangle(xy, width, height, color = 'red', linewidth = 1, fill = False, alpha = None, axis = None):
    """
    draw a rectangle on plt axis
    """
    import matplotlib.patches as patches
    rect = patches.Rectangle(
        xy = xy,
        width = width,
        height = height,
        alpha = alpha,
        color = color,
        fill = fill,
        linewidth = linewidth
    )
    if axis is not None:
        axis.add_patch(rect)
    return rect
    
rect = rectangle

def maximize_figure():
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()

def line(xy_start, xy_end, color = 'red', linewidth = 1, alpha = None, axis = None):
    """
    draw a line on plt axis
    """
    from matplotlib.lines import Line2D 
    num = 100
    xdata = np.linspace(xy_start[0], xy_end[0], num = num)
    ydata = np.linspace(xy_start[1], xy_end[1], num = num)
    line = Line2D(
        alpha = alpha,
        color = color,
        linewidth = linewidth,
        xdata = xdata,
        ydata = ydata
    )
    if axis is not None:
        axis.add_line(line)
    return line

def imshow(title = None, img = None, gray = False):
    show_images([img], [title], gray = gray)

def show_images(images, titles = None, shape = None, share_axis = False, 
                bgr2rgb = False, maximized = False, 
                show = True, gray = False, save = False, colorbar = False, 
                path = None, axis_off = False, vertical = False, subtitle = None):
        
    if shape == None:
        if vertical:
            shape = (len(images), 1)
        else:
            shape = (1, len(images))
        
    ret_axes = []
    ax0 = None
    for idx, img in enumerate(images): 
        if bgr2rgb:
            img = util.img.bgr2rgb(img)
        loc = (idx / shape[1], idx % shape[1])
        if idx == 0:
            ax = plt.subplot2grid(shape, loc)
            ax0 = ax
        else:
            if share_axis:
                ax = plt.subplot2grid(shape, loc, sharex = ax0, sharey = ax0)
            else:
                ax = plt.subplot2grid(shape, loc)
        if len(np.shape(img)) == 2 and gray:
            img_ax = ax.imshow(img, cmap = 'gray')
        else:
            img_ax = ax.imshow(img)
        
        if len(np.shape(img)) == 2 and colorbar:
            plt.colorbar(img_ax, ax = ax)
        if titles != None:
            ax.set_title(titles[idx])
        
        if axis_off:
            plt.axis('off')
#             plt.xticks([]), plt.yticks([])
        ret_axes.append(ax)
        
    if subtitle is not None:
        set_subtitle(subtitle)
    if maximized:
        maximize_figure()
        
    if save:
        if path is None:
            raise ValueError('path can not be None when save is True')
        save_image(path)
    if show:
        plt.show()
    return ret_axes

def save_image(path, img = None, dpi = 150):
    path = util.io.get_absolute_path(path)
    util.io.make_parent_dir(path)
    if img is None:
        plt.gcf().savefig(path, dpi = dpi)
    else:
        plt.imsave(path, img)

imwrite = save_image
    
def to_ROI(ax, ROI):
    xy1, xy2 = ROI
    xmin, ymin = xy1
    xmax, ymax = xy2
    ax.set_xlim(xmin, xmax)
    #ax.extent
    ax.set_ylim(ymax, ymin)
    
def set_subtitle(title, fontsize = 12):
    plt.gcf().suptitle(title, fontsize=fontsize)

def show(maximized = False):
    if maximized:
        maximize_figure()
    plt.show()
    
def draw():
    plt.gcf().canvas.draw()

def get_random_line_style():
    colors = ['r', 'g', 'b']
    line_types = ['-']#, '--', '-.', ':']
    idx = util.rand.randint(len(colors))
    color = colors[idx]
    idx = util.rand.randint(len(line_types))
    line_type = line_types[idx]
    return color + line_type
