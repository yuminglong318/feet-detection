import numpy as np
import os

def iou(box, clusters):
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return (np.max(iou_), np.argmax(iou_))

def avg_iou(boxes, clusters):
    return np.mean([iou(boxes[i], clusters)[0] for i in range(boxes.shape[0])])

def kmeans(boxes, k, dist=np.median):
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)[0]

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

def load_dataset(directory): 
    dataset = [] 
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename)) as f:
            if filename.endswith('.txt'):
                tokens = f.read().split()
                
                width = float(tokens[3])
                height = float(tokens[4])
                dataset.append([width, height]) 
    return np.array(dataset) 

n_clusters = 6 # change this if you wish to try other number of centroids
directory = './'

data = load_dataset(directory)
out = kmeans(data, k=n_clusters)
print(out* 640)