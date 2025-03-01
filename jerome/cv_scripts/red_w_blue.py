import cv2
import numpy as np

def replace_red_with_natural_blue(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return
    
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper bounds for red color in HSV
    lower_red1 = np.array([0, 80, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 80, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks to detect red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2
    
    # Extract the red regions
    red_regions = cv2.bitwise_and(image, image, mask=red_mask)
    
    # Convert red regions to blue shades while preserving natural texture
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(image_lab)
    
    # Further adjust color channels to achieve a more natural blueberry look
    a[red_mask > 0] = np.clip(a[red_mask > 0] - 100, 0, 255)  # Further reduce redness
    b[red_mask > 0] = np.clip(b[red_mask > 0] + 140, 0, 255)  # Stronger blueness boost
    """
    """
    
    # Merge modified channels back
    modified_lab = cv2.merge((l, a, b))
    result = cv2.cvtColor(modified_lab, cv2.COLOR_LAB2BGR)
    
    # Save and display the output
    cv2.imwrite(output_path, result)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
replace_red_with_natural_blue("test.jpg", "output.jpg")

