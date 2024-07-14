import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
import cv2

# Ensure TensorFlow does not use all GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def load_and_preprocess_image(image_path, target_size=(1280, 960)):
    """Load and preprocess a TIFF image."""
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = img_to_array(img)
    img = img / 255.0  # Normalize to [0, 1] range
    return img

def apply_kmeans_to_image(image, n_clusters=2):
    """Apply k-means clustering to an image."""
    h, w, c = image.shape
    reshaped_image = image.reshape(-1, c)  # Flatten the image
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(reshaped_image)
    clustered_image = kmeans.cluster_centers_[kmeans.labels_]
    clustered_image = clustered_image.reshape(h, w, c)
    return clustered_image

def count_cells(clustered_image, min_size=10000):
    """Count cells in the clustered image."""
    if clustered_image.shape[-1] != 3:
        clustered_image = cv2.cvtColor(clustered_image, cv2.COLOR_GRAY2RGB)

    # Convert the clustered image to grayscale
    gray = cv2.cvtColor((clustered_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on size and circularity
    cell_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_size:
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            #if 0.7 < circularity < 1.3:  # Adjust circularity threshold as needed
            cell_contours.append(contour)

    # Count the cells
    cell_count = len(cell_contours)

    # Draw contours for visualization (optional)
    output_image = (clustered_image * 255).astype(np.uint8)
    cv2.drawContours(output_image, cell_contours, -1, (0, 255, 0), 2)

    # Save the output image with contours drawn
    cv2.imwrite('output_image.png', output_image)

    return cell_count, output_image

# Example usage
image_path = 'C:\\Users\\humza\\OneDrive\\Desktop\\Job\\Segmentation\\USH-Week42-_0080.tif'

# Load and preprocess the image
image = load_and_preprocess_image(image_path)

# Apply k-means clustering
clustered_image = apply_kmeans_to_image(image)

# Count the cells in the clustered image
cell_count, output_image = count_cells(clustered_image, min_size=0)
print(f'Number of cells: {cell_count}')

# Plotting the original, clustered, and output images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(np.squeeze(image))

plt.subplot(1, 3, 2)
plt.title('Clustered Image')
plt.imshow(np.squeeze(clustered_image))

plt.subplot(1, 3, 3)
plt.title(f'Output Image\nNumber of cells: {cell_count}')
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))

plt.show()
