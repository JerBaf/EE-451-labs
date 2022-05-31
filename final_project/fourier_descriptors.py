import numpy as np
import matplotlib.pyplot as plt
import cv2
from helpers import *


DIRECTIONS = [(0,-1),(1,0),(0,1),(-1,0),(-1,-1),(1,-1),(1,1),(-1,1)]

def plot_im(image,cmap="gray"):
    fig,ax= plt.subplots(figsize=(16,9))
    ax.imshow(image,cmap=cmap)

def has_neighbours(im,pt):
    return im[pt[0]-1:pt[0]+2,pt[1]-1:pt[1]+2].sum() > 1

def get_new_pt(tmp_im):
    zero_idx = np.where(tmp_im > 0)
    actual_pt = (-1,-1)
    if zero_idx[0].shape[0] != 0: # Still some pixels left
        actual_pt = (zero_idx[0][0],zero_idx[1][0])
    return actual_pt

def get_next_pt(im,pt):
    searching = True
    i = 0
    for i in range(len(DIRECTIONS)):
        d = DIRECTIONS[i]
        new_pt = (pt[0]+d[0],pt[1]+d[1])
        if (new_pt[0] < 0 or new_pt[0] >= im.shape[0] or
                new_pt[1] < 0 or new_pt[1] >= im.shape[1]):
            pass
        elif im[new_pt] > 0:
            return new_pt
    return pt

def order_contour(im):
    tmp_im = np.copy(im)
    tmp_pt = get_new_pt(tmp_im)
    contour = []
    while tmp_im.sum() != 0 and tmp_pt != (-1,-1):
        if has_neighbours(tmp_im,tmp_pt):
            contour.append(tmp_pt)
            tmp_im[tmp_pt] = 0
            tmp_pt = get_next_pt(tmp_im,tmp_pt)
        else:
            tmp_im[tmp_pt] = 0
            tmp_pt = get_new_pt(tmp_im)
    return contour

def get_fourier_descriptors_alt(contour,n=2):
    """ Compute the fourier descriptors of a digit. """
    complex_contour = list(map(lambda l: complex(l[0],l[1]), contour))
    fourier_coeffs = fft(complex_contour)
    return np.absolute(fourier_coeffs[1:n+1])


def extract_features(image,n=2):
    ret, thresholded_im = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresholded_im[thresholded_im == 0] = 1
    thresholded_im[thresholded_im == 255] = 0
    thresholded_im = median_filter(thresholded_im)
    contour_points,_ = cv2.findContours(thresholded_im, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(image.shape,np.uint8)
    for i in contour_points:
        for j in i:
            mask[j[0][1],j[0][0]] = 1
    contour = order_contour(mask)
    descriptors = get_fourier_descriptors_alt(contour,n=n)
    return descriptors