import cv2
import numpy as np
import os


def load_net(cfg_file= 'models/cfg/yolov3-tiny-arch.cfg', weights_file= 'models/weights/yolov3-tiny-arch_final.weights'):

    net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)

    return net

def detect_feet(image_file, net):

    # Load the image
    image = cv2.imread(image_file)

    # height and width of image
    height, width, _ = image.shape

    layer_names = net.getLayerNames()
    output_layers = net.getUnconnectedOutLayersNames()

    # Prepare the image for inference
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (640, 640), (0,0,0), True, crop=False)

    # Set the new input value for the network
    net.setInput(blob)

    # Run a forward pass
    outs = net.forward(output_layers)

    # detected feet list
    feet = []

    for out in outs: 
        for detection in out: 
            # Each detection has the x, y, w and h normalized (from 0 to 1) relative to the width and the height of the image
            scores = detection[5:] 
            class_id = np.argmax(scores) 
            confidence = scores[class_id]

            if confidence > 0.2:

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                feet.append({'confidence': confidence, 'cx': center_x, 'cy': center_y, 'width': w, 'height': h})

    return feet

def draw_bounding_box(image_file, feet):

    # Load the image
    image = cv2.imread(image_file)

    # height and width of image
    height, width, _ = image.shape

    if(len(feet) == 0):
        return image

    else:

        center_x = sum(f['cx'] for f in feet) / len(feet)
        center_y = sum(f['cy'] for f in feet) / len(feet)
        w = sum(f['width'] for f in feet) / len(feet)
        h = sum(f['height'] for f in feet) / len(feet)

        # foot = max(feet, key = lambda e: e['confidence'])
        # center_x = foot['cx']
        # center_y = foot['cy']
        # w = foot['width']
        # h = foot['height']
        
        # Rectangle coordinates
        x = int(center_x - w / 2)
        y = int(center_y - h / 2)

        # Ensure the bounding boxes are within the bounds of the image
        x = max(min(x, width - 1), 0)
        y = max(min(y, height - 1), 0)
        w = int(min(width - x - 1, w))
        h = int(min(height - y - 1, h))

        # Draw the bounding box on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return image


if __name__ == '__main__':

    # cfg_file = 'models/cfg/yolov3-tiny-feet.cfg'                            # Your .cfg file
    # weights_file = 'models/weights/yolov3-tiny-feet_final.weights'          # Your .weights file
    
    net = load_net()

    image_directory = './feet_data-arch/valid'                        # Input image

    for image_name in os.listdir(image_directory):

        if image_name.endswith('.jpg'):
            image_file = os.path.join(image_directory, image_name)

            detections = detect_feet(image_file, net)

            image = draw_bounding_box(image_file, detections)

            cv2.imwrite(f"./results-arch/{image_name}", image)