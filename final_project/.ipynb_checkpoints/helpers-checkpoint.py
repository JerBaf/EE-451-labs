import cv2
import numpy as np
import k_means
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import scipy.signal as signal
from skimage.color import rgb2gray
from sklearn import cluster
from PIL import Image,ImageEnhance


# HELPERS

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

def segment_table(table_im_big):
    reduction_factor = 30 #30
    target_size = (int(table_im_big.shape[1]/reduction_factor),
                   int(table_im_big.shape[0]/reduction_factor))
    table_im = cv2.resize(table_im_big,target_size)
    
    gray_sobel = cv2.cvtColor(sobel_filter(table_im,balance=1), cv2.COLOR_BGR2GRAY)
    log = LoG(gray_sobel,sigma=1,tr=20)
    
    ### Detect center
    kernel = np.ones((7,7))
    inverse_log = -log + 1
    start_mask = cv2.morphologyEx(inverse_log, cv2.MORPH_OPEN, kernel)
    x_center, y_center = start_mask.shape[0]/2,start_mask.shape[1]/2
    start_indices = zip(np.where(start_mask == 1)[0],np.where(start_mask == 1)[1])
    start = sorted(start_indices,key=lambda l: np.power((l[0]-x_center,l[1]-y_center),2).sum())[0]
    
    ### Segment table
    kernel = np.ones((3,3))
    inverse_log = cv2.morphologyEx(inverse_log, cv2.MORPH_OPEN, kernel)
    segmentation = np.zeros(inverse_log.shape,dtype=np.uint8)
    for i in iterative_grow(inverse_log,start[0],start[1],0.5):
        segmentation[i[0],i[1]] = 255
    corners = retrieve_corners_alt(segmentation)
    table_segmentation = extract_table_alt(table_im_big,corners,reduction_factor)

    return table_segmentation

def extract_T_cards(table_segmentation, table_canny):
    table_rotated = cv2.rotate(table_segmentation, cv2.cv2.ROTATE_90_CLOCKWISE)
    canny = table_canny
    
    # Params
    bottom_boundary = 2600
    T = 500
    horizontal_buffer = 50
    vertical_buffer = 50

    # Cut out bottom of image with supposed right-cards
    T_cards = table_segmentation[bottom_boundary:-1,:]
    canny = canny[bottom_boundary:-1,:]    

    # Get card boundaries
    cards_horizontal_sum = np.sum(canny, axis=0)
    left_boundary = np.min(np.where(cards_horizontal_sum > T)) - horizontal_buffer
    right_boundary = np.max(np.where(cards_horizontal_sum > T)) + horizontal_buffer
    diff = right_boundary - left_boundary
    split = int(diff/5)

    # Separate cards roughly
    cards = []
    for i in range(5):
        cur_low = left_boundary + i*split
        cur_high = cur_low + split
        canny_cut = canny[:,cur_low:cur_high]
        canny_vertical_sum = np.sum(canny_cut, axis=1)
        upper_boundary = np.min(np.where(canny_vertical_sum > T)) - vertical_buffer
        lower_boundary = np.max(np.where(canny_vertical_sum > T)) + vertical_buffer
        if upper_boundary < 0:
            upper_boundary = 0
        if lower_boundary > canny_cut.shape[0]:
            lower_boundary = canny_cut.shape[0]
        cards.append(T_cards[upper_boundary:lower_boundary,cur_low:cur_high])
        
    return cards

def extract_right_cards(table_segmentation, table_canny):
    table_rotated = cv2.rotate(table_segmentation, cv2.cv2.ROTATE_90_CLOCKWISE)
    canny = cv2.rotate(table_canny, cv2.cv2.ROTATE_90_CLOCKWISE)
    
    # Params
    bottom_boundary = 2600
    T = 500
    horizontal_buffer = 50
    vertical_buffer = 50

    # Cut out bottom of image with supposed right-cards
    right_cards = table_rotated[bottom_boundary:-1,1100:2300]
    canny = canny[bottom_boundary:-1,1100:2300]

    # Get card boundaries
    cards_horizontal_sum = np.sum(canny, axis=0)
    left_boundary = np.min(np.where(cards_horizontal_sum > T)) - horizontal_buffer
    right_boundary = np.max(np.where(cards_horizontal_sum > T)) + horizontal_buffer
    if left_boundary < 0:
        left_boundary = 0
    if right_boundary > canny.shape[1]:
        right_boundary = canny.shape[1]

    # Exctract cards roughly
    cards = []
    canny = canny[:,left_boundary:right_boundary]
    canny_vertical_sum = np.sum(canny, axis=1)
    upper_boundary = np.min(np.where(canny_vertical_sum > T)) - vertical_buffer
    lower_boundary = np.max(np.where(canny_vertical_sum > T)) + vertical_buffer
    if upper_boundary < 0:
        upper_boundary = 0
    if lower_boundary > canny.shape[0]:
        lower_boundary = canny.shape[0]
    cards.append(right_cards[upper_boundary:lower_boundary,left_boundary:right_boundary])
    
    return cards

def extract_left_cards(table_segmentation, table_canny):
    table_rotated = cv2.rotate(table_segmentation, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    canny = cv2.rotate(table_canny, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Params
    bottom_boundary = 2600
    T = 500
    horizontal_buffer = 50
    vertical_buffer = 50

    # Cut out bottom of image with supposed right-cards
    left_cards = table_rotated[bottom_boundary:-1,1100:2300]
    canny = canny[bottom_boundary:-1,1100:2300]
    
    # Get card boundaries
    cards_horizontal_sum = np.sum(canny, axis=0)
    left_boundary = np.min(np.where(cards_horizontal_sum > T)) - horizontal_buffer
    right_boundary = np.max(np.where(cards_horizontal_sum > T)) + horizontal_buffer
    if left_boundary < 0:
        left_boundary = 0
    if right_boundary > canny.shape[1]:
        right_boundary = canny.shape[1]

    # Exctract cards roughly
    cards = []
    canny = canny[:,left_boundary:right_boundary]
    canny_vertical_sum = np.sum(canny, axis=1)
    upper_boundary = np.min(np.where(canny_vertical_sum > T)) - vertical_buffer
    lower_boundary = np.max(np.where(canny_vertical_sum > T)) + vertical_buffer
    if upper_boundary < 0:
        upper_boundary = 0
    if lower_boundary > canny.shape[0]:
        lower_boundary = canny.shape[0]
    cards.append(left_cards[upper_boundary:lower_boundary,left_boundary:right_boundary])
    
    return cards

def extract_top_cards(table_segmentation, table_canny):
    table_rotated = cv2.rotate(table_segmentation, cv2.cv2.ROTATE_180)
    canny = cv2.rotate(table_canny, cv2.cv2.ROTATE_180)
    sides = [canny[:,0:int(canny.shape[1]/2)], canny[:,int(canny.shape[1]/2):-1]]

    # Params
    bottom_boundary = 2625
    T = 500
    horizontal_buffer = 50
    vertical_buffer = 50

    cards = []
    for idx, side in enumerate(sides):
        # Cut out bottom of image with supposed right-cards
        card_im = side[bottom_boundary:-1,250:1300]

        # Get card boundaries
        cards_horizontal_sum = np.sum(card_im, axis=0)
        left_boundary = np.min(np.where(cards_horizontal_sum > T)) - horizontal_buffer
        right_boundary = np.max(np.where(cards_horizontal_sum > T)) + horizontal_buffer
        if left_boundary < 0:
            left_boundary = 0
        if right_boundary > card_im.shape[1]:
            right_boundary = card_im.shape[1]

        # Exctract cards roughly
        cards_vertical_sum = np.sum(card_im, axis=1)
        upper_boundary = np.min(np.where(cards_vertical_sum > T)) - vertical_buffer
        lower_boundary = np.max(np.where(cards_vertical_sum > T)) + vertical_buffer
        if upper_boundary < 0:
            upper_boundary = 0
        if lower_boundary > card_im.shape[0]:
            lower_boundary = card_im.shape[0]

        if idx == 1:
            shift_x = int(canny.shape[1]/2)
            cards.append(table_rotated[upper_boundary + bottom_boundary:lower_boundary + bottom_boundary,
                                       left_boundary + shift_x + 250:right_boundary + shift_x + 250])
        else:
            cards.append(table_rotated[upper_boundary + bottom_boundary:lower_boundary + bottom_boundary,
                                       left_boundary + 250:right_boundary + 250])
            
    return cards

def find_chips(chips, r_min=120, r_max=130):
    all_chips = chips.copy() 
    
    gray = cv2.cvtColor(all_chips, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, r_min, param1=85, param2=15, minRadius=r_min, maxRadius=r_max)
    
    masks = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            one_chip = chips.copy()
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(all_chips, center, radius, (0, 0, 0), thickness=-1)
            cv2.circle(one_chip, center, radius, (0, 0, 0), thickness=-1)
            gray_chip = rgb2gray(one_chip)
            mask = (gray_chip < 0.0001).astype(int)
            masks.append(mask)

    gray2 = rgb2gray(all_chips)
    big_mask = (gray2 < 0.0001)
    
    return big_mask, masks

def get_brightness(img):
    gray_img = rgb2gray(img)
    measure = np.median(gray_img)
    return measure


def get_chips_labels(table_segmentation, plot=False):
    brightness = get_brightness(table_segmentation)
    chips = table_segmentation[1300:2500,1100:2500]
    mask_all, masks = find_chips(chips)

    all_chips = chips.copy()
    all_chips[mask_all==0] = 0

    g = cluster.KMeans(n_clusters=5).fit(chips[(mask_all == 1)]) 

    for msk in masks:
        pred = g.predict(chips[msk == 1])
        one_chip = chips.copy()
        one_chip[msk == 0] = 0
        
    labels = np.array([(pred==0).sum(), (pred==1).sum(), (pred==2).sum(), (pred==3).sum(), (pred==4).sum()])
    g.cluster_centers_ = np.array([[245., 213., 193.], [64., 38., 21.], [177., 99., 8.], [112., 89., 7.], [64., 43., 131.]])
    
    img = Image.fromarray(table_segmentation)
    img_brightness_obj = ImageEnhance.Brightness(img)
    factor = brightness/get_brightness(table_segmentation) # (1 + brightness/get_brightness(table_segmentation))/2
    enhanced_img = img_brightness_obj.enhance(factor)
    
    chips = np.array(enhanced_img)[800:2700,800:2700]
    mask_all, masks = find_chips(chips)
    if plot: 
        plt.imshow(chips)
        plt.show()
        all_chips = chips.copy()
        all_chips[mask_all==0] = 0
        plt.imshow(all_chips, cmap='gray')
        plt.show()
    labels = np.array([])
    for msk in masks:
        pred = g.predict(chips[msk == 1])
        labels = np.append(labels, np.argmax(np.array([(pred==0).sum(), (pred==1).sum(), (pred==2).sum(), (pred==3).sum(), 
                                                       (pred==4).sum()])))
    return labels
