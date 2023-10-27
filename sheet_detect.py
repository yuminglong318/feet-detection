import cv2
import numpy as np
import os
from feet_detect_top import load_net, detect_feet

net = load_net()
for image_file in os.listdir('feet_data-top/train/'):
    if image_file.endswith('.jpg'):
        # Read the image in grayscale
        image = cv2.imread(os.path.join('feet_data-top/train/', image_file))

        feet = detect_feet(os.path.join('feet_data-top/train/', image_file), net)

        center_x = sum(f['cx'] for f in feet) / len(feet)
        center_y = sum(f['cy'] for f in feet) / len(feet)
        w = sum(f['width'] for f in feet) / len(feet)
        h = sum(f['height'] for f in feet) / len(feet)

        f_x1 = int(center_x - w/2)
        f_y1 = int(center_y - h/2)
        f_x2 = int(center_x + w/2)
        f_y2 = int(center_y + h/2)

        # Calculate the region coordinates
        if h > w:
            x1 = int(center_x - w/1.5)
            y1 = int(center_y - 10)
            x2 = int(center_x + w/1.5)
            y2 = int(center_y + 10)
        else:
            x1 = int(center_x - 10)
            y1 = int(center_y - h/1.5)
            x2 = int(center_x + 10)
            y2 = int(center_y + h/1.5)
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # # Apply Gaussian blur
        # blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # # Apply Canny edge detection
        # edges = cv2.Canny(gray_image, 100, 150)

        # Apply Sobel operator
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        edges = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
        
        gray_image = cv2.equalizeHist(gray_image)
        # Apply the mask to the gray image
        # gray_image = cv2.bitwise_or(gray_image, mask)

        gray_image = cv2.subtract(gray_image, edges)
        
        # gray_image[y1:y2, x1:x2] = 255
        
        _, thresholded_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

        # Create a 5x5 kernel for maximum filtering
        kernel = np.ones((5, 20), np.uint8)

        # Apply maximum filtering to the binary image
        filtered_image = cv2.erode(thresholded_image, kernel)

        # Set pixels to 0 where the filtered image is different from the original binary image
        modified_image = np.where(filtered_image != thresholded_image, 0, thresholded_image)

        modified_image[f_y1:f_y2, f_x1:f_x2] = 0
        modified_image[y1:y2, x1:x2] = 255

        # cv2.rectangle(modified_image, (x1, y1), (x2, y2), 0, thickness=2)

        contours, _ = cv2.findContours(modified_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if w >= 240 and h >= 240:
                # print(x,y,w,h)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imwrite(f'results-sheet/{image_file}', image)
        cv2.imwrite(f'temp/{image_file}', modified_image)
