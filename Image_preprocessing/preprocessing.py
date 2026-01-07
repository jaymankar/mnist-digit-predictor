import numpy as np 
import re, base64, io, torch
from PIL import Image

def preprocess_image(base64_str: str) -> torch.Tensor:
    # Decode base64
    match = re.search(r"base64,(.*)", base64_str)
    if not match:
        raise ValueError("Invalid image data")
    
    base64_img = match.group(1)
    img_decode = base64.b64decode(base64_img)
    img_file_like = io.BytesIO(img_decode)
    
    # Open image as grayscale
    img = Image.open(img_file_like).convert('L')
    
    # Convert to NumPy array
    img_arr = np.array(img)
    
    # Crop to content
    coords = np.column_stack(np.where(img_arr < 255))  # pixels that are not white
    if coords.size > 0:
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        img_arr = img_arr[x_min:x_max+1, y_min:y_max+1]
    else:
        img_arr = np.zeros((28,28), dtype=np.uint8)
    
    # Resize to 20x20
    img = Image.fromarray(img_arr)
    img.thumbnail((20,20), Image.Resampling.LANCZOS)
    
    # Place into 28x28 canvas
    new_img = Image.new('L', (28,28), color=255)  # white background
    paste_x = (28 - img.width) // 2
    paste_y = (28 - img.height) // 2
    new_img.paste(img, (paste_x, paste_y))
    
    # Convert to float32 and normalize 0-1
    img_arr = np.array(new_img).astype('float32') / 255.0
    
    # Invert colors: black pen → white digit on black background
    img_arr = 1.0 - img_arr
    
    # Convert to PyTorch tensor (1,1,28,28)
    img_tensor = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0)
    
    return img_tensor
