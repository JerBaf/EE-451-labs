import cv2
import numpy as np


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