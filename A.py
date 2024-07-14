import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
import cv2

def load_and_preprocess_image(image_path, target_size=(1280, 960)):
    """Load and preprocess a TIFF image."""
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img)
    if len(img.shape) == 2:  # Check if the image is grayscale
        img = np.stack((img,)*3, axis=-1)  # Convert to RGB by duplicating channels
    img = img / 255.0  # Normalize to [0, 1] range
    return img

def apply_kmeans_to_image(image, n_clusters=2):
    """Apply k-means clustering to an image."""
    h, w, c = image.shape
    reshaped_image = image.reshape(-1, c)  # Flatten the image
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)  # Explicitly set n_init
    kmeans.fit(reshaped_image)
    clustered_image = kmeans.cluster_centers_[kmeans.labels_]
    clustered_image = clustered_image.reshape(h, w, c)
    return clustered_image

def binary_threshold_image(clustered_image):
    """Apply simple thresholding to the clustered image and return the binary image."""
    if clustered_image.shape[-1] != 3:
        clustered_image = cv2.cvtColor(clustered_image, cv2.COLOR_GRAY2RGB)

    # Convert the clustered image to grayscale
    gray = cv2.cvtColor((clustered_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Apply simple thresholding (Otsu's method)
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_image

def apply_morphological_operations(binary_image):
    """Apply morphological operations to the binary image to remove noise and close gaps."""
    # Define a structuring element for opening (remove small noise)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_open)

    # Define a structuring element for closing (close gaps)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel_close)

    return closed_image

def count_large_white_areas(binary_image, min_size=10000):
    """Count the number of large white areas in the binary image."""
    # Perform connected component labeling
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # Filter out small components
    large_area_count = 0
    large_areas = np.zeros_like(labels, dtype=np.uint8)
    for i in range(1, num_labels):  # Start from 1 to skip the background
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_size:
            large_area_count += 1
            large_areas[labels == i] = 255
    
    return large_area_count, large_areas

def crop_diagnostic_info(image, crop_height=50):
    """Crop out the diagnostic information at the bottom of the image."""
    return image[:-crop_height, :, :]

# Load and preprocess the image
image_path = 'C:\\Users\\humza\\OneDrive\\Desktop\\Job\\Segmentation\\USH-Week42-_0080.tif'
image = load_and_preprocess_image(image_path)

cropped_image = crop_diagnostic_info(image, crop_height=65)

# Apply k-means clustering
clustered_image = apply_kmeans_to_image(cropped_image)

# Get the binary image
binary_image = binary_threshold_image(clustered_image)

# Apply morphological operations to the binary image
closed_image = apply_morphological_operations(binary_image)

# Count the large white areas
large_area_count, large_areas_image = count_large_white_areas(closed_image, min_size=5000)
print(f'Number of large white areas: {large_area_count}')

plt.imsave('cropped_image.png', cropped_image)
plt.imsave('clustered_image.png', clustered_image)
plt.imsave('binary_image.png', binary_image, cmap='gray')
plt.imsave('closed_image.png', closed_image, cmap='gray')
plt.imsave('large_areas_image.png', large_areas_image, cmap='gray')

# Plotting the original, binary, and large areas images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(np.squeeze(image))

plt.subplot(1, 3, 2)
plt.title('Binary Image')
plt.imshow(binary_image, cmap='gray')

plt.subplot(1, 3, 3)
plt.title(f'Large Areas\nCount: {large_area_count}')
plt.imshow(large_areas_image, cmap='gray')

plt.show()
