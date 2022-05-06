import numpy as np

# HELPERS
def initialize_clusters(data, k):
    """randomly initialize the k cluster centers (the means). Make sure you choose k clusters from the data itself,
             or ensure otherwise that your initializations have the same scale as the data
    
    Args:
        data: shape = (N, d). original data. 
        k: integer number. predefined number of clusters for the k-means algorithm. 
    Returns:
        numpy array with shape (k, d) which corresponds to the k initial clusters.
    """
    N, _ = data.shape
    return data[np.random.choice(N, k)]

def build_distance_matrix(data, mu):
    """builds a distance matrix.
    
    Args:
        data: numpy array of shape = (N, d). original data. 
        mu:   numpy array of shape = (k, d). Each row corresponds to a cluster center.
    Returns:
        squared distances matrix,  numpy array of shape (N, k):
            row number i column j corresponds to the squared distance of datapoint i with cluster center j.
    """
    N, _ = data.shape
    k, _ = mu.shape
    distance_matrix = np.zeros((N, k))
    for j in range(k):
        distance_matrix[:, j] = np.sum(np.square(data - mu[j, :]), axis=1)
    return distance_matrix


def update_kmeans_parameters(data, mu_old):
    """compute one step of the kmeans algorithm: using mu_old, find to which cluster each datapoint belongs to, 
            then update the parameter cluster centers.
    
    Args:
        data:   numpy array of shape = (N, d). original data. 
        mu_old: numpy array of shape = (k, d). Each row corresponds to a cluster center.
    Returns:
        losses: shape (N, ), contains the (old) squared distances of each data point to its (old) cluster mean (computed from mu_old).
        assignments: vector of shape (N, ) which contains the cluster associated to each data point.
        mu: updated vector mu of shape (k, d) where each row corresponds to the new cluster center.
    """
    _, d = data.shape
    k, _ = mu_old.shape
    distance_matrix = build_distance_matrix(data, mu_old)
    losses = np.min(distance_matrix, axis=1)
    assignments = np.argmin(distance_matrix, axis=1)

    # update the mu
    mu = np.zeros((k, d))
    for j in range(k):
        rows = np.where(assignments == j)[0]
        mu[j, :] = np.mean(data[rows, :], axis=0)
    return losses, assignments, np.nan_to_num(mu)


def preprocess_image(original_image,dim=1):
    """preprocess the image. 
    vectorize the three matrices (each matrix corresponds to a RGB color channel). **don't normalize!** 
    
    Args: 
        original_image: numpy array of shape (480, 480, 3) 
    Returns:
        processed_image: numpy array of shape (480*480, 3)
    """
    processed_image = original_image.reshape(
        (original_image.shape[0] * original_image.shape[1], dim))
    processed_image = processed_image.astype(float)
    print(
        "Current image: the shape of image={s}, the data type={dt}.".format(
            s=processed_image.shape, dt=processed_image.dtype))
    return processed_image

def kmean_compression(original_image, k=3, max_iters=100, threshold=1e-7):
    """using k-means for image compression.
    Args: 
        original_image: numpy array of shape (480, 480, 3).
        processed_image: numpy array of shape (480*480, 3).
        k: scalar. Number of clusters.
        max_iters: integer. Max number of iterations for the kmeans algorithm.
        threshold: scalar. Stop the kmeans algorithm if the loss decrease between two iterations
                        is lower than the threshold.
    """
    
    processed_image = preprocess_image(original_image,3)
    mu_old = initialize_clusters(processed_image, k)
    
    # init some empty lists to store the result.
    loss_list = []
    
    # start the kmeans
    for iter in range(max_iters):
        losses, assignments, mu = update_kmeans_parameters(processed_image, mu_old)
        
        # calculate the average loss over all points
        average_loss = np.mean(losses)
        loss_list.append(average_loss)

        if iter % 10 == 0:
            print(
                "The current iteration of kmeans is: {i}, the average loss is {l}.".format(
                    i=iter, l=average_loss))
        
        # check converge
        if iter > 0 and np.abs(loss_list[-1] - loss_list[-2]) < threshold:
            break

        # update mu
        mu_old = mu

    image_reconstruct = mu[assignments].reshape(original_image.shape).astype(np.uint8)
    return image_reconstruct

def aggregate(img,x,y,color):
    candidate = []
    region = []
    visited = set()
    candidate.append((x,y))
    if not in_image(img,x,y):
        raise Exception("The seed given is not in the image boundaries.")
    while len(candidate) > 0:
        c = candidate[-1]
        x_c, y_c = c
        if img[x_c,y_c] == color and (x_c,y_c) not in visited:
            visited = visited.union(set([tuple((x_c,y_c))]))
            region.append(c)
            candidate.pop()
            for new_c in neighbours(img,x_c,y_c):
                candidate.append(new_c)
        else:
            candidate.pop()
    return region

def neighbours(img,x,y):
    neighbours = []
    for i in range(-1,2):
        for j in range(-1,2):
            if not (i==0 and j==0):
                if in_image(img,x+i,y+j):
                    neighbours.append((x+i,y+j))
    return neighbours

def in_image(img,x,y):
    return x >= 0 and y >= 0 and x < img.shape[0] and y < img.shape[1]