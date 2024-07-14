import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import label
from PIL import Image

# Load the provided clustered image
clustered_image_path = 'H9-Week42-_0033.tif'
clustered_image = Image.open(clustered_image_path)
clustered_image = np.array(clustered_image)

# Display the loaded clustered image
plt.imshow(clustered_image)
plt.title('Clustered Image')
plt.show()

# Ensure the clustered image is in grayscale if not already
if clustered_image.ndim == 3:
    gray_image = cv2.cvtColor(clustered_image, cv2.COLOR_RGB2GRAY)
else:
    gray_image = clustered_image

# Check the pixel intensity range of the grayscale image
pixel_min = gray_image.min()
pixel_max = gray_image.max()
print(f'Pixel intensity range: {pixel_min} to {pixel_max}')

# Normalize the grayscale image to the range [0, 255] if necessary
if pixel_max <= 1.0:
    gray_image = (gray_image * 255).astype(np.uint8)

# Apply a binary threshold to get a binary image
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Perform connected component analysis to count cells
labeled_image, num_features = label(binary_image)

# Plotting the grayscale, binary, and labeled images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Grayscale Image')
plt.imshow(gray_image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Binary Image')
plt.imshow(binary_image, cmap='gray')

plt.subplot(1, 3, 3)
plt.title(f'Labeled Image - Cells Count: {num_features}')
plt.imshow(labeled_image, cmap='nipy_spectral')

plt.show()

print(f'Number of cells detected: {num_features}')
