import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from remap import *

'''def mask_color(image, color):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define range of yellow color in HSV
    if (color == "yellow"):
        lower = np.array([10, 40, 100])
        upper = np.array([30, 255, 255])
        # Threshold the HSV image to get only yellow colors
        mask = cv2.inRange(hsv, lower, upper)
        # Bitwise-AND mask and original image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
    elif (color == "red"):
        lower = np.array([0, 100, 50])  # Capture deep reds while avoiding dull colors
        upper = np.array([10, 255, 255])  # Capture bright reds
        lower2 = np.array([160, 80, 40])  # Capture dark reds in the upper hue range
        upper2 = np.array([180, 255, 255])  # Capture deep red tones
        mask1 = cv2.inRange(hsv, lower, upper)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        masked_image = cv2.bitwise_or(mask1, mask2)
    return masked_image'''

def mask_color(image, color):

    if color == "yellow":
        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Define range of yellow color in HSV
        lower_yellow = np.array([10, 40, 100])
        upper_yellow = np.array([30, 255, 255])
        # Threshold the HSV image to get only yellow colors
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # Bitwise-AND mask and original image
    elif color == "red":
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Define RGB range for red
        lower = np.array([0, 0, 0])     # Lower bound for deep reds
        upper = np.array([255, 60, 150]) # Upper bound for bright reds
        # Create the mask
        mask = cv2.inRange(rgb, lower, upper)

    # Apply mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def detect_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)     # Use Canny edge detection
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=10, minLineLength=100, maxLineGap=10)
    return lines

def draw_lines(image, lines):
    image_with_lines = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image_with_lines

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image

def convert_and_dilate(image, kernel_size=(5, 5)):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones(kernel_size, np.uint8)
    dilated_image = cv2.dilate(gray_image, kernel, iterations=5)
    
    return gray_image, dilated_image


def contours(image, filename):
    
    # Find Canny edges
    edged = cv2.Canny(image, 100, 200)

    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Sort contours by area and top bottom difference 2in descending order 
    # helps distinguish the ears of corn vs misdetections
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]  # Select the four largest contours
    contours = sorted(contours, key=lambda x: x[:, :, 1].max() - x[:, :, 1].min(), reverse=True)[:2]

    ret = np.array(np.zeros((2,4)))

    for idx, contour in enumerate(contours):
        # Calculate highest and lowest points of contour
        ext_top = tuple(contour[contour[:, :, 1].argmin()][0])
        ext_bot = tuple(contour[contour[:, :, 1].argmax()][0])
        cv2.drawContours(image, [contour], -1, (255, 255, 0), thickness=10)

        # cv2.imshow("im", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Draw dots on highest and lowest points
        cv2.circle(image, ext_top, 5, (0, 0, 255), -1)
        cv2.circle(image, ext_bot, 5, (0, 0, 255), -1)

        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Draw bounding rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), thickness=10)

        # Draw horizontal lines connecting sides of bounding box with contour
        horizontal_distances = []
        for i in range(y, y + h + 1, int(h / 10)):
            point1 = (x, i)
            point2 = (x + w, i)
            cv2.line(image, point1, point2, (0, 255, 0), thickness=2)

            # Find intersection points of the line with the contour
            intersections = []
            for pt in contour:
                if pt[0][1] == i:
                    if x <= pt[0][0] <= x + w:
                        intersections.append((pt[0][0], pt[0][1]))

            # Draw circles at intersection points
            for intersection in intersections:
                cv2.circle(image, intersection, 5, (255, 0, 0), 10)

            # Calculate horizontal distances
            if len(intersections) > 1:
                horizontal_distances.append(abs(intersections[0][0] - intersections[-1][0]))


        # Draw vertical lines connecting sides of bounding box with contour
        vertical_distances = []
        for i in range(x, x + w + 1, int(w / 10)):
            point1 = (i, y)
            point2 = (i, y + h)
            cv2.line(image, point1, point2, (0, 255, 0), thickness=2)

            # Find intersection points of the line with the contour
            intersections = []
            for pt in contour:
                if pt[0][0] == i:
                    if y <= pt[0][1] <= y + h:
                        intersections.append((pt[0][0], pt[0][1]))

            # Draw circles at intersection points
            for intersection in intersections:
                cv2.circle(image, intersection, 5, (255, 0, 0), 20)

            # Calculate vertical distances
            if len(intersections) > 1:
                vertical_distances.append(abs(intersections[0][1] - intersections[-1][1]))

        # cv2.imshow("intersections", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite('detections/detect_' + str(filename), image)

        # Calculate average width and height based on horizontal and vertical distances
        avg_width = np.mean(horizontal_distances) if horizontal_distances else 0
        avg_height = np.mean(vertical_distances) if vertical_distances else 0

        ret[idx, 0] = avg_width
        ret[idx, 1] = w
        ret[idx, 2] = avg_height
        ret[idx, 3] = h

    return ret

def contours(image, filename):
    # Find Canny edges
    edged = cv2.Canny(image, 100, 200)

    # Find contours
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Sort contours by area (largest first) and select only the largest one
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]  # Select the single largest contour

    ret = np.zeros((1, 4))  # Store width and height metrics for one contour

    for idx, contour in enumerate(contours):
        # Calculate highest and lowest points of the contour
        ext_top = tuple(contour[contour[:, :, 1].argmin()][0])
        ext_bot = tuple(contour[contour[:, :, 1].argmax()][0])
        cv2.drawContours(image, [contour], -1, (255, 255, 0), thickness=10)

        # Draw dots on highest and lowest points
        cv2.circle(image, ext_top, 5, (0, 0, 255), -1)
        cv2.circle(image, ext_bot, 5, (0, 0, 255), -1)

        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Draw bounding rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), thickness=10)

        plt.figure(figsize=(10, 8))
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.gca().add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none'))
        plt.title(f"Bounding Box Visualization for {filename}")
        plt.axis('off')
        plt.show()


        # Draw horizontal lines connecting bounding box sides with the contour
        horizontal_distances = []
        for i in range(y, y + h + 1, int(h / 10)):
            point1 = (x, i)
            point2 = (x + w, i)
            cv2.line(image, point1, point2, (0, 255, 0), thickness=2)

            # Find intersection points
            intersections = [tuple(pt[0]) for pt in contour if pt[0][1] == i and x <= pt[0][0] <= x + w]

            # Draw intersection circles
            for intersection in intersections:
                cv2.circle(image, intersection, 5, (255, 0, 0), 1)

            # Calculate width distances
            if len(intersections) > 1:
                horizontal_distances.append(abs(intersections[0][0] - intersections[-1][0]))

        # Draw vertical lines connecting bounding box sides with the contour
        vertical_distances = []
        for i in range(x, x + w + 1, int(w / 10)):
            point1 = (i, y)
            point2 = (i, y + h)
            cv2.line(image, point1, point2, (0, 255, 0), thickness=2)

            # Find intersection points
            intersections = [tuple(pt[0]) for pt in contour if pt[0][0] == i and y <= pt[0][1] <= y + h]

            # Draw intersection circles
            for intersection in intersections:
                cv2.circle(image, intersection, 5, (255, 0, 0), 1)

            # Calculate height distances
            if len(intersections) > 1:
                vertical_distances.append(abs(intersections[0][1] - intersections[-1][1]))

        cv2.imwrite('detections/detect_' + str(filename), image)

        # Calculate average width and height based on horizontal and vertical distances
        avg_width = np.mean(horizontal_distances) if horizontal_distances else 0
        avg_height = np.mean(vertical_distances) if vertical_distances else 0

        ret[0, 0] = avg_width
        ret[0, 1] = w
        ret[0, 2] = avg_height
        ret[0, 3] = h

    return ret

'''def contours(image, filename):
    # Find Canny edges
    edged = cv2.Canny(image, 100, 200)

    # Find contours
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Sort contours by area (largest first) and select only the largest one
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]  # Select the single largest contour

    ret = np.zeros((1, 4))  # Store width and height metrics for one contour

    for idx, contour in enumerate(contours):
        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Draw bounding rectangle using OpenCV (optional for saving the image)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), thickness=10)

        # Visualize bounding box using Matplotlib
        plt.figure(figsize=(10, 8))
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.gca().add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none'))
        plt.title(f"Bounding Box Visualization for {filename}")
        plt.axis('off')
        plt.show()

        # Calculate average width and height based on horizontal and vertical distances
        avg_width = w
        avg_height = h

        ret[0, 0] = avg_width
        ret[0, 1] = w
        ret[0, 2] = avg_height
        ret[0, 3] = h

    return ret'''

def main():

    # Define the folder containing the images
    folder_path = 'Popcorn Images/Ears'
    image_data = []

    color = input("Enter the color of the corn: ")

    # Loop through the images in the folder
    for idx, filename in enumerate(os.listdir(folder_path)):
       
        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)
        cv2.imwrite("unprocessed_image.png", img)
        height, width = img.shape[:2]

        if width > height:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) 
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title("Unprocessed Image")
        plt.axis('off')
        plt.show()

        try:
            scaled_img = scale_img(img)
        except:
            print("An error occured")
            continue

        color_img = mask_color(scaled_img, color)
        img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title("Color Mask")
        plt.axis('off')
        plt.show()
        
        
        gray_image, dilated_image = convert_and_dilate(color_img)
        img_rgb = cv2.cvtColor(dilated_image, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title("Dilated Image")
        plt.axis('off')
        plt.show()
        
        
        blurred =  cv2.medianBlur(dilated_image, 51)
        img_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title("Blurred Image")
        plt.axis('off')
        plt.show()
        
        
        data = contours(blurred, filename)
        # print(data.shape[0])

        # Create a dictionary with image information
        for i in range(data.shape[0]): 
            image_info = {
                "name": str(filename),
                "avg_width": data[i][0],
                "max_width": data[i][1],
                "avg_height": data[i][2],
                "max_height": data[i][3],
                "max_width_cm": (data[i][1] / 390) * 19.5,
                "max_height_cm": (data[i][3] / 490) * 24.5,
            }
            image_data.append(image_info)


        print("Finished image: " + str(filename))

    # Write the image data to a JSON file
    output_file = 'image_data.json'
    with open(output_file, 'w') as json_file:
        json.dump(image_data, json_file, indent=4)

    print(f"Image data has been written to {output_file} successfully.")

if __name__ == "__main__":
    main()