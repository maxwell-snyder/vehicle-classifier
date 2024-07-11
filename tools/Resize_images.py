from PIL import Image
import os

def resize_images(input_dir, output_dir, size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            
            # Convert the image to RGB mode if it's in palette mode
            if img.mode == 'P':
                img = img.convert('RGB')
            
            img = img.resize(size, Image.LANCZOS)  # Use Image.LANCZOS instead of Image.ANTIALIAS
            img.save(os.path.join(output_dir, filename))

input_dir = r"C:\Users\max\OneDrive\My Coding Projects\VS code\projects\Vehicle detction\vehicle images old\train\truck"
output_dir = r"C:\Users\max\OneDrive\My Coding Projects\VS code\projects\Vehicle detction\dataset\train\truck"
size = (150, 150)

resize_images(input_dir, output_dir, size)


