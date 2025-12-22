import cv2
import numpy as np
import matplotlib.pyplot as plt

def pencil_sketch(image_path, blur_kernel=75):
    """
    Convert an image to pencil sketch effect.

    Args:
    image_path (str): Path to input image
    blur_kernel (int): Gaussian blur kernel size (must be odd)

    Returns:
    tuple: (original_rgb, sketch) or (None, None) if error
    """
    # TODO: Implement the algorithm
    # Step 1: Load image
    original=cv2.imread(image_path)
    # Step 2: Convert to grayscale
    gray=cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    # Step 3: Invert grayscale
    inverted=255 - gray
    # Step 4: Apply Gaussian blur
    blurred=cv2.GaussianBlur(inverted, (blur_kernel, blur_kernel), 0)
    # Step 5: Invert blurred image
    inverted_blur=255 - blurred
    # Step 6: Divide and scale
    sketch=np.minimum(255,(gray/inverted_blur)*256)
    sketch=sketch.astype(np.uint8)
    original_rgb=cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    return original_rgb,sketch
    pass    
def display_result(original, sketch, save_path=None):
    """
    Display original and sketch side-by-side.

    Args:
    original: Original image (RGB)
    sketch: Sketch image (grayscale)
    save_path: Optional path to save the sketch
    """
    # TODO: Create matplotlib figure with 1 row, 2 columns
    # Display original on left, sketch on right
    # Add titles and remove axes
    images=[original,sketch]
    titles=['Original','Sketch']
    fig,axes=plt.subplots(1,2,figsize=(8,8))
    for ax,img,title in zip(axes,images,titles):
        ax.imshow(img,cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.show()
    else:
        plt.show()
    pass
def main():
    """Main function to run the pencil sketch converter."""
    # TODO: Get image path from user or command line
    # Call pencil_sketch function
    # Call display_result function
    # Handle any errors
    image_path=input("Enter the path to the image: ")
    original,sketch=pencil_sketch(image_path)
    display_result(original,sketch,'sketch_output.png')
    pass
if __name__ == '__main__':
   main()