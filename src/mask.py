from PIL import Image
import numpy as np

def create_quadrant_mask():
    # Create a 1024x1024 white image
    image = Image.new('RGB', (1024, 1024), 'white')
    
    # Convert to numpy array for easier manipulation
    img_array = np.array(image)
    
    # Set the top-left quadrant (512x512) to black
    img_array[:512, :512] = [0, 0, 0]
    
    # Convert back to PIL Image
    result = Image.fromarray(img_array)
    
    # Save the image
    result.save('quadrant_mask.png')
    
    return result

if __name__ == '__main__':
    create_quadrant_mask()
