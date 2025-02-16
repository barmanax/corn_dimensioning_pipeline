# Corn Ear Detection and Processing

This project processes images to detect and measure ears of corn. It performs various image processing tasks like masking, edge detection, and line drawing. The results are stored in a JSON file containing information about detected contours. The "Detections" folder shows the result of the algorithm on each input image. The "charts" folder contains example visualizations of the data.

## Requirements

I run the script with the following software installed: 

- Python 3.11.4
- OpenCV 4.8.0
- NumPy 1.24.3
- Matplotlib 3.7.1

You can install the libraries on mac like so:

```bash
# Install OpenCV 4.8.0
pip install opencv-python==4.8.0

# Install NumPy 1.24.3
pip install numpy==1.24.3

# Install Matplotlib 3.7.1
pip install matplotlib==3.7.1
```

To process the images in Popcorn Images/Ears, run main.py. This will generate the json file image_data.json and populate the detections folder. 

```bash
python main.py
```

To generate charts based off image_data.json, run charts.py. This will repopulate the charts folder. 

```bash
python charts.py
```
