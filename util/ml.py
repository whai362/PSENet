import logging
import cv2
import numpy as np
import util.dec
import util.np

@util.dec.print_calling
def kmeans(samples, k, criteria = None, attempts = 3, flags = cv2.KMEANS_RANDOM_CENTERS):
    if criteria == None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    samples = np.asarray(samples, dtype = np.float32)
    _,labels,centers = cv2.kmeans(samples, k, criteria, attempts, flags)
    labels = util.np.flatten(labels)
    clusters = [None]*k
    for idx, label in enumerate(labels):
        if clusters[label] is None:
            clusters[label] = []
        clusters[label].append(idx)
        
    for  idx, cluster in enumerate(clusters):
        if cluster == None:
            logging.warn('Empty cluster appeared.')
            clusters[idx] = []
            
    return labels, clusters, centers
