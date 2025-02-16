import cv2
import matplotlib.pyplot as plt

# Read the image from the folder
image = cv2.imread('C:/Users/adity/OneDrive/Documents/corn_ear_detection/Popcorn Images/Ears_Test/IMG_9520.jpeg')  # Replace with the actual path and file name

# Convert the image from BGR (OpenCV default) to RGB (Matplotlib uses RGB)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image using Matplotlib
plt.imshow(image_rgb)
plt.axis('off')  # Turn off axis for better visualization
plt.title('Image Display')  # Optional: Add a title
plt.show()