from PIL import Image
import os

def resize_images(input_dir, output_dir, size):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        img = Image.open(os.path.join(input_dir, filename))
        img = img.resize(size)
        img.save(os.path.join(output_dir, filename))

if __name__ == "__main__":
    input_dir = r"C:\Users\max\OneDrive\My Coding Projects\VS code\projects\Vehicle detction\Vehicle images old\\truck"
    output_dir = r"C:\Users\max\OneDrive\My Coding Projects\VS code\projects\Vehicle detction\Vehicle images\\truck"
    size = (150, 150)  # Adjust size as needed
    resize_images(input_dir, output_dir, size)