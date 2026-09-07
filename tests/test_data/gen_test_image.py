import numpy as np
import tifffile
import os

def generate_validation_image():
    # Dynamically pull the exact max value for a 16-bit unsigned integer (65535)
    UINT16_MAX = np.iinfo(np.uint16).max
    GRAY_50 = UINT16_MAX // 2  # Evaluates to 32767
    
    # 1920x1080 (HD) layout for easy dividing
    width = 1920
    height = 1080
    
    # Initialize a 16-bit RGB array with zeros (pure black)
    image = np.zeros((height, width, 3), dtype=np.uint16)

    # ---------------------------------------------------------
    # 1. Continuous Grayscale Ramp (Top Half: Rows 0 to 540)
    # ---------------------------------------------------------
    # Create a 1D gradient array from 0 to UINT16_MAX exactly matching the width
    gradient_1d = np.linspace(0, UINT16_MAX, width, dtype=np.uint16)
    
    # Apply the gradient across the top half of the image for all 3 channels (RGB)
    image[0:540, :, 0] = gradient_1d
    image[0:540, :, 1] = gradient_1d
    image[0:540, :, 2] = gradient_1d

    # ---------------------------------------------------------
    # 2. 50% Neutral Gray Block (Middle Quarter: Rows 540 to 810)
    # ---------------------------------------------------------
    image[540:810, :] = GRAY_50

    # ---------------------------------------------------------
    # 3. Pure RGB & CMY Blocks (Bottom Quarter: Rows 810 to 1080)
    # ---------------------------------------------------------
    # 6 blocks total. 1920 / 6 = exactly 320 pixels wide per block.
    block_w = width // 6

    # Define the 16-bit colors using the constant
    colors = [
        (UINT16_MAX, 0, 0),                   # Red
        (0, UINT16_MAX, 0),                   # Green
        (0, 0, UINT16_MAX),                   # Blue
        (0, UINT16_MAX, UINT16_MAX),          # Cyan
        (UINT16_MAX, 0, UINT16_MAX),          # Magenta
        (UINT16_MAX, UINT16_MAX, 0)           # Yellow
    ]

    for i, color in enumerate(colors):
        x_start = i * block_w
        x_end = (i + 1) * block_w
        image[810:1080, x_start:x_end] = color

    # ---------------------------------------------------------
    # Save the File
    # ---------------------------------------------------------
    filename = 'reference_color_grading_input.tiff'
    
    # Save as an uncompressed, mathematically lossless 16-bit RGB TIFF
    tifffile.imwrite(filename, image, photometric='rgb', compression='none')
    
    print(f"Validation image successfully generated: {os.path.abspath(filename)}")

if __name__ == "__main__":
    generate_validation_image()
    