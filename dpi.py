from PIL import Image

def get_image_metadata(image_path):
    with Image.open(image_path) as img:
        dpi = img.info.get('dpi', None)
        if dpi:
            print(f"DPI: {dpi}")
        else:
            print("DPI information not available")

        return dpi
    

image_path = r"C:\Users\adity\OneDrive\Documents\corn_ear_detection\Popcorn Images\Ears\38687.JPG"
dpi = get_image_metadata(image_path)

