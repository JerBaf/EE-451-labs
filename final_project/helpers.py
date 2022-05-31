import cv2
import numpy as np
import k_means
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import scipy.signal as signal


#HELPERS

def LoG(img,sigma=1,tr=150):
    gaussian = cv2.GaussianBlur(img,(0,0),sigma)
    log = cv2.Laplacian(gaussian, cv2.CV_64F)
    edge_mask = zero_crossing(log,tr)
    return edge_mask

def zero_crossing(img,tr=0):
    zero_crossing = np.zeros(img.shape,dtype=np.float32)
    max_diff = np.abs(img.max() - img.min())
    for i in range(1,img.shape[0]):
        for j in range(1,img.shape[1]):
            local_window = img[i-1:i+2,j-1:j+2]
            local_min = local_window.min()
            local_max = local_window.max()
            if local_min < 0 and local_max > 0 and (local_max - local_min) > tr:
                zero_crossing[i,j] = 1
    return zero_crossing

def sobel_filter(img,balance=0.2):
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    final = cv2.addWeighted(grad, balance, img, 1-balance, 0)
    return final

def median_filter(img,k=3):
    new_image = np.zeros(img.shape,dtype=np.uint8)
    offset = int((k-1)/2)
    for i in range(offset,img.shape[0]-offset):
        for j in range(offset,img.shape[1]-offset):
            new_image[i,j] = int(np.median(img[i-offset:i+offset+1,j-offset:j+offset+1]))
    return new_image

def retrieve_corners_opt(mask):
    indices = np.where(mask > 0)
    top_sort = sorted(zip(indices[1]-indices[0],range(indices[0].shape[0])),key=lambda l: l[0])
    right_sort = sorted(zip(indices[1]+indices[0],range(indices[0].shape[0])),key=lambda l: l[0])
    top_idx = top_sort[-1]
    bot_idx = top_sort[0]
    right_idx = right_sort[-1]
    left_idx = right_sort[0]
    top = (indices[0][top_idx[1]],indices[1][top_idx[1]])
    right = (indices[0][right_idx[1]],indices[1][right_idx[1]])
    bot = (indices[0][bot_idx[1]],indices[1][bot_idx[1]])
    left = (indices[0][left_idx[1]],indices[1][left_idx[1]])
    return (top, right, bot, left)

def retrieve_corners_alt(mask):
    indices = np.where(mask > 0)
    bot_sort = sorted(zip(indices[0],range(indices[0].shape[0])),key=lambda l: l[0])
    right_sort = sorted(zip(indices[1],range(indices[0].shape[0])),key=lambda l: l[0])
    top_idx = bot_sort[0]
    bot_idx = bot_sort[-1]
    right_idx = right_sort[-1]
    left_idx = right_sort[0]
    y_max = bot_idx[0]
    y_min = top_idx[0]
    x_max = right_idx[0]
    x_min = left_idx[0]
    return (y_max, y_min, x_max, x_min)

def extract_table_alt(original_im,corners,reduction_factor):
    y_max, y_min, x_max, x_min = corners
    y_max = y_max*reduction_factor
    y_min = y_min*reduction_factor
    x_max = x_max*reduction_factor
    x_min = x_min*reduction_factor
    return original_im[y_min:y_max,x_min:x_max]

def extract_table(original_im,corners,reduction_factor):
    top, right, bot, left = corners
    # Get corners
    top_idx = reduction_factor * top[0]
    bot_idx = reduction_factor * bot[0]
    right_idx = reduction_factor * right[1]
    left_idx = reduction_factor * left[1]
    # Rotate image if needed
    c_t, c_r, c_b, c_l = corners
    tan_c = (c_l[0]-c_t[0])/(c_t[1]-c_l[1])
    angle_c = np.arctan(tan_c)
    angle_c = np.ceil(angle_c*180/np.pi)
    if angle_c > 3:
        h,w,_ = original_im.shape
        image_center = (w//2,h//2)
        M = cv2.getRotationMatrix2D(center=(image_center),angle=-angle_c,scale=1)
        rotated = cv2.warpAffine(original_im,M,(w,h))
        rotated = rotated[top_idx:bot_idx,left_idx:right_idx,:]
        return rotated
    # Retrieve table
    table_im = original_im[top_idx:bot_idx,left_idx:right_idx,:]
    return table_im

def in_image(img,x,y):
    return x >= 0 and y >= 0 and x < img.shape[0] and y < img.shape[1]

def neighbours(img,x,y):
    neighbours = []
    for i in range(-1,2):
        for j in range(-1,2):
            if not (i==0 and j==0):
                if in_image(img,x+i,y+j):
                    neighbours.append((x+i,y+j))
    return neighbours

def iterative_grow(img,x,y,tr):
    candidate = []
    region = []
    visited = set()
    candidate.append((x,y))
    if not in_image(img,x,y):
        raise Exception("The seed given is not in the image boundaries.")
    while len(candidate) > 0:
        c = candidate[-1]
        x_c, y_c = c
        if img[x_c,y_c] > tr and not ((x_c,y_c) in visited):
            visited = visited.union(set([tuple((x_c,y_c))]))
            region.append(c)
            candidate.pop()
            for new_c in neighbours(img,x_c,y_c):
                candidate.append(new_c)
        else:
            candidate.pop()
    return region


def create_k_mean_mask(image):
    if len(image.shape) > 2: 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    compressed_im = k_means.kmean_compression(gray,k=2)
    mask = np.zeros(compressed_im.shape,dtype=np.uint8)
    indices_above_mean = np.where(compressed_im > compressed_im.mean())
    indices_below_mean = np.where(compressed_im <= compressed_im.mean())
    if indices_above_mean[0].shape[0] > indices_below_mean[0].shape[0]:
        mask[indices_below_mean] = 255
    else:
        mask[indices_above_mean] = 255
    return mask

def shape_detector(image):
    im = image.copy()
    indices = np.where(im > 0)
    shape_masks = []
    while indices[0].shape[0] > 0:
        x,y = indices[0][0],indices[1][0]
        shape = iterative_grow(im,x,y,0.5)
        shape_mask = np.zeros(im.shape,dtype=np.uint8)
        for i in shape:
            shape_mask[i[0],i[1]] = 255
            im[i[0],i[1]] = 0
        # update
        if len(shape) > 50:
            print(len(shape))
            shape_masks.append(shape_mask)
        indices = np.where(im > 0)
    return shape_masks



    ### CONTOURS

directions = np.asarray([[1,-1],[1,1],[-1,1],[-1,-1],[0,-1],[1,0],[0,1],[-1,0]]) # Trigonometric wise

def get_contour(im):
    """ Apply a linear filter to retrieve the contour points. """
    contours, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_mask = np.zeros(im.shape)
    for i in contours:
        for j in i:
            contour_mask[j[0][1],j[0][0]] = 1
    sharp_contour_mask = sharp_edges(contour_mask)
    return sharp_contour_mask

def get_ordered_contour(im):
    """ Get the outer border of a digit. """
    contour_mask = get_contour(im)
    point = np.asarray([np.where(contour_mask == 1)[0][0],np.where(contour_mask == 1)[1][0]])
    contour = set([tuple(point)])
    id = 0
    contour_id = [(tuple(point),id)] # will be used to order correctly the contour
    new_point = True
    while new_point: 
        new_point = False
        for move in directions:
            candidate = point+move
            if contour_mask[candidate[0],candidate[1]] == 1 and tuple(candidate) not in contour:
                new_point = True
                id += 1
                contour = contour.union(set([tuple(candidate)]))
                contour_id.append((tuple(candidate),id))
                point = candidate
                break
    return list(map(lambda l : l[0], sorted(contour_id,key=lambda t : t[1])))

def sharp_edges(im):
    """ Remove the stairs part in a contour mask. """
    kernel_left_stairs = np.asarray([[0,1,0],[1,1,0],[0,0,0]]).astype(np.uint8)
    left_stairs = cv2.morphologyEx(im,
                                  cv2.MORPH_ERODE,kernel=kernel_left_stairs)
    kernel_right_stairs = np.asarray([[0,1,0],[0,1,1],[0,0,0]]).astype(np.uint8)
    right_stairs = cv2.morphologyEx(im-left_stairs,
                                   cv2.MORPH_ERODE,kernel=kernel_right_stairs)
    kernel_up_stairs = np.asarray([[0,0,0],[0,1,1],[0,1,0]]).astype(np.uint8)
    up_stairs = cv2.morphologyEx(im-left_stairs-right_stairs,
                                cv2.MORPH_ERODE,kernel=kernel_up_stairs)
    kernel_down_stairs = np.asarray([[0,0,0],[1,1,0],[0,1,0]]).astype(np.uint8)
    down_stairs = cv2.morphologyEx(im-left_stairs-right_stairs-up_stairs,
                                  cv2.MORPH_ERODE,kernel=kernel_down_stairs)
    return im - left_stairs - right_stairs - up_stairs - down_stairs

def complex_contour(im):
    """ Get the contour of the image in complex number form and ordered. """
    contour = get_ordered_contour(im)
    complex_contour = list(map(lambda l: complex(l[0],l[1]), contour))
    return complex_contour 

def show_contour(im):
    """ Plot the contour of the digit. """
    contour = get_ordered_contour(im)
    mask = np.zeros(im.shape)
    for pt in contour:
        mask[pt] = 1
    fig,ax = plt.subplots(1,2)
    ax[0].set_title("Digit's Contour")
    ax[1].set_title("Digit")
    ax[0].imshow(mask,cmap = 'gray')
    ax[1].imshow(im,cmap="gray")

def get_fourier_descriptors(im,n=2):
    """ Compute the fourier descriptors of a digit. """
    imaginary_contour = complex_contour(im)
    fourier_coeffs = fft(imaginary_contour)
    return fourier_coeffs[1:n+1]