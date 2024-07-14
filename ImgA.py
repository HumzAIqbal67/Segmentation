import cv2
import numpy as np
import matplotlib.pyplot as plt

def count_cells(image_path, min_size=1000):
    # Step 1: Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Step 3: Adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Step 4: Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Filter contours based on size and circularity
    cell_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_size:
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 0.7 < circularity < 1.3:  # Adjust circularity threshold as needed
                cell_contours.append(contour)

    # Step 6: Count the cells
    cell_count = len(cell_contours)

    # Draw contours for visualization (optional)
    output_image = image.copy()
    cv2.drawContours(output_image, cell_contours, -1, (0, 255, 0), 2)

    # Save the output image with contours drawn
    cv2.imwrite('output_image.png', output_image)

    return cell_count, output_image

# Example usage
image_path = 'C:\\Users\\humza\\OneDrive\\Desktop\\Job\\Segmentation\\USH-Week42-_0080.tif'
cell_count, output_image = count_cells(image_path, min_size=10)
print(f'Number of cells: {cell_count}')

# Display the output image with contours (optional)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title(f'Number of cells: {cell_count}')
plt.show()
