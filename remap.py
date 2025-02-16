import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def filter_black_areas(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define the range for black color in HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 102])  # Adjust the upper bound as needed, original was [180, 255, 130]
    # Create a mask for black areas
    black_mask = cv2.inRange(hsv_image, lower_black, upper_black)
    # cv2.imshow("black mask", black_mask)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return black_mask

def find_sheet(image):

    # filter for green (paper) in hsv space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    
    lower_green = np.array([10, 50, 100])
    upper_green = np.array([100, 255, 255])
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    yellow_parts = cv2.bitwise_and(image, image, mask=mask)

    # filter for green (paper) in rgb space
    lower_green = np.array([0, 100, 50])
    upper_green = np.array([100, 255, 200])
    mask = cv2.inRange(image, lower_green, upper_green)
    green_parts = cv2.bitwise_and(image, image, mask=mask)
    
    # combine the two filters
    both_mask = cv2.bitwise_or(yellow_parts, green_parts)
    cv2.imwrite("both_mask.png", both_mask)

    gray_image = cv2.cvtColor(both_mask, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("gray_image.png", gray_image)

    # find the contours
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    if contours:
        largest_contour = max(contours, key=cv2.contourArea) # the page will be the largest contour in the filtered image
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        filtered_image = cv2.bitwise_and(image, image, mask=mask)
        img_rgb = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)
        '''
        plt.imshow(img_rgb)
        plt.title("Isolated Green Page")
        plt.axis('off')
        plt.show()
        '''
        return filtered_image
    else:
        print("No contours found.")
        return None

'''def find_corner_coords(img):
    # Step 1: Filter black areas (assuming filter_black_areas is defined elsewhere
    black_mask = filter_black_areas(img)
    plt.imshow(black_mask)
    plt.axis('off')
    plt.show()
    # Step 2: Apply morphological operations to refine the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    refined_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)

    # plt.imshow(refined_mask)
    # plt.axis('off')
    # plt.show()

    # Step 3: Smooth the image to reduce noise
    blurImg = cv2.GaussianBlur(refined_mask, (15, 15), 0)

    # Step 4: Find contours
    contours, _ = cv2.findContours(blurImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    print(len(contours))
    # Step 5: Filter and store valid contours
    contours_with_area = []
    min_area = 10
    max_area = img.shape[0] * img.shape[1] * 0.2  # 20% of the image area

    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)

        if num_vertices == 6:  # Look for shapes with 6 vertices
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                contours_with_area.append((approx, area))
    # Step 6: Sort the contours and find the best 4
    sorted_contours = sorted(contours, key=lambda x: x[1])  # Sort by area


    if len(contours) < 4:
        raise ValueError("Not enough corners detected. Check lighting or adjust parameters.")

    best_subset = []
    min_diff = float("inf")

    for i in range(len(sorted_contours) - 3):
        current_subset = sorted_contours[i:i + 4]
        current_areas = [area for _, area in current_subset]
        current_diff = current_areas[-1] - current_areas[0]
        if current_diff < min_diff:
            min_diff = current_diff
            best_subset = current_subset

    # Step 7: Calculate the centroids of the best 4 corners
    sheet_coordinates = np.empty((0, 2), dtype="float32")

    for contour, _ in best_subset:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        sheet_coordinates = np.vstack([sheet_coordinates, [cx, cy]])

    # Step 8: Verify and return the coordinates
    if len(sheet_coordinates) != 4:
        raise ValueError("Failed to find exactly 4 corners. Verify input image and parameters.")

    return sheet_coordinates
'''

def find_corner_coords(img) : 

    black_mask = filter_black_areas(img)
    img_rgb = cv2.cvtColor(black_mask, cv2.COLOR_BGR2RGB)
    '''
    plt.imshow(img_rgb)
    plt.title("Image with Black Mask")
    plt.axis('off')
    plt.show()
    '''
    kernel_size=(2, 2)
    kernel = np.ones(kernel_size, np.uint8)
    blurImg = cv2.blur(black_mask, (10,10))  # helps to smooth contour detections (not over detect)
   
    contours, hierarchy = cv2.findContours(image=blurImg, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # Define circle parameters
    circle_radius = 20
    text_color = (255, 0, 255) 

    # list to store contours that meet corner requirements
    contours_with_area = []

    min_area = 1000
    w, h, d = img.shape
    area_sheet = w*h

    # Loop through contours and filter by vertices
    for contour in contours:
        epsilon = 0.01  * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)  # Approximate the contour to a polygon
        num_vertices = len(approx)
        
        if num_vertices == 6: # the page corner markers have 6 vertices 
            area = cv2.contourArea(contour)
            if  area > min_area and area < (area_sheet*0.2): # approximation for size of corner
                contours_with_area.append((approx, area))
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                else:
                    cx, cy = 0, 0

                    # text = f"Vertices: {len(approx)}"
                    # cv2.putText(img, text, (cx - 60, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 1, cv2.LINE_AA)
                
                text = f" Area: {area}"
                cv2.putText(img, text, (cx - 60, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 5 , cv2.LINE_AA)
                cv2.drawContours(image=img, contours=[contour], contourIdx=-1, color=(0, 255, 0), thickness=4, lineType=cv2.LINE_AA)

    # cv2.imshow("blur", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
                
    w, h, d = img.shape
    area_sheet = w*h
   
    # Sort the contours by size
    sorted_contours = sorted(contours_with_area, key=lambda x: x[1])

    # for i in sorted_contours :  print(i[1])
    # print("# of detections: " +  str(len(sorted_contours)))

    min_diff = float('inf')
    best_subset = []

    # sliding window approach to find the closest group of 4 (the four corners are closest in area)
    for i in range(len(sorted_contours) - 3): 
        current_subset = sorted_contours[i:i + 4] 
        # print(str(i) + " - " + str(i+4))
        current_areas = [area for _, area in current_subset]
        current_diff = current_areas[-1] - current_areas[0]
        # print(current_areas)
        # print(current_diff) 
        if abs(current_diff) < abs(min_diff) :
            min_diff = current_diff
            best_subset = current_subset

    # print('\n ')


    sheet_coordinates = np.empty((0, 2), dtype="float32")  # Initialize as an empty 2D array

    for contour, area in best_subset:

        cv2.drawContours(image=img, contours=[contour], contourIdx=-1, color=(255, 255, 0), thickness=4, lineType=cv2.LINE_AA)

        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0

        # Draw a circle at the centroid of the contour
        cv2.circle(img, (cx, cy), circle_radius, (255, 0, 0), -1)
        sheet_coordinates = np.vstack([sheet_coordinates, [cx, cy]])
 
        
    # Show the images
    # cv2.imshow("Detections", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return sheet_coordinates


def transform_sheet(image, corners):
    for corner in corners:
       cv2.circle(image, (int(corner[0]), int(corner[1])), radius=10, color=(0, 0, 255), thickness=-1)

    # Display image with detected corners
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    '''
    plt.imshow(img_rgb)
    plt.title("Page with corners mapped")
    plt.axis('off')
    plt.show()
    '''

    # Define the correct aspect ratio (24.5 cm x 19.5 cm)
    output_width = 390  # Scale 24.5 cm to 490 pixels
    output_height = 490  # Scale 19.5 cm to 390 pixels

    # Define the four destination points to match the bounding box ratio
    pts_dst = np.array([[0, 0], [output_width - 1, 0], [0, output_height - 1], [output_width - 1, output_height - 1]], dtype="float32")

    sorted_corners = np.empty((0, 2), dtype="float32")

    # Match each corner to the correct destination
    for point_src in pts_dst:
        distances = np.linalg.norm(corners - point_src, axis=1)
        closest_index = np.argmin(distances)
        sorted_corners = np.vstack([sorted_corners, corners[closest_index]])
        corners = np.delete(corners, closest_index, 0)

    offset = 50
    sorted_corners = sorted_corners + np.array([[-offset, -offset], [offset, -offset], [-offset, offset], [offset, offset]])

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(np.float32(sorted_corners), np.float32(pts_dst))

    # Apply the perspective transformation with the corrected aspect ratio
    warped_image = cv2.warpPerspective(image, M, (output_width, output_height))
    '''
    plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    plt.title("Warped Image")
    plt.axis('off')
    plt.show()
    '''
    return warped_image

def scale_img(img) : 
    sheet_filtered = find_sheet(img)
    corners = find_corner_coords(sheet_filtered)
    warped = transform_sheet(img, corners)
    img_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    return warped


if __name__ == "__main__":
    # Define the folder containing the images
    folder_path = 'C:/Users/adity/OneDrive/Documents/corn_ear_detection/Popcorn Images/Ears_Test'

    # Initialize an empty list to store image information

    # Loop through the images in the folder
    for idx, filename in enumerate(os.listdir(folder_path)):
    
        image_path = os.path.join(folder_path, filename)
        # image_path =   "Popcorn Images/Ears/IMG_9737.jpeg"
        img = cv2.imread(image_path)
        # find_coords(img)

        sheet_filtered = find_sheet(img)

        # cv2.imshow("sheet", sheet_filtered)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        corners = find_corner_coords(sheet_filtered)
        warped = transform_sheet(img, corners)

        # cv2.imshow("warped", warped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # if idx == 10 :  
        #     cv2.destroyAllWindows()
        #     break
    
    # img = cv2.imread("Popcorn Images/Ears/IMG_9721.jpeg")
    # sheet_filtered = find_sheet(img)
    # find_corner_coords(sheet_filtered)


