import cv2
import numpy as np
import os

# Files
cfg_file = 'models/cfg/yolov3-tiny-feet.cfg'                            # Your .cfg file
weights_file = 'models/weights/yolov3-tiny-feet_final.weights'               # Your .weights file
classes = ['feet']                                           # Your classes file
image_directory = './feet_data/train'                        # Input image

# with open(classes_file, 'r') as f:
#     classes = f.read().split('\n')

# Load the network
net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)

for image_name in os.listdir(image_directory):

    if image_name.endswith('.jpg'):
        image_file = os.path.join(image_directory, image_name)

        # Load the image
        image = cv2.imread(image_file)

        height, width, _ = image.shape

        # Get the output layer names
        layer_names = net.getLayerNames()
        output_layers = net.getUnconnectedOutLayersNames()

        # Prepare the image for inference
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (640, 640), (0,0,0), True, crop=False)

        # Set the new input value for the network
        net.setInput(blob)

        # Run a forward pass
        outs = net.forward(output_layers)

        count = 0
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


        if(len(feet) == 0):
            print(image_name)

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
            cv2.putText(image, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        # # Display the image
        # cv2.imshow("Image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite(f"./results/{image_name}", image)