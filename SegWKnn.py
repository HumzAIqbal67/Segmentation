import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
import os

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
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(reshaped_image)
    clustered_image = kmeans.cluster_centers_[kmeans.labels_]
    clustered_image = clustered_image.reshape(h, w, c)
    return clustered_image

# Example usage
image_path = 'USH-Week42-_0080.tif'

# Load and preprocess the image
image = load_and_preprocess_image(image_path)

# Apply k-means clustering
clustered_image = apply_kmeans_to_image(image)

# Plotting the original and clustered image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(np.squeeze(image))

plt.subplot(1, 2, 2)
plt.title('Clustered Image')
plt.imshow(np.squeeze(clustered_image))

# Maybe reduce interations/increase threshold to lower the actual clustering precision. Will be easier to detect connectedness.
# Read some papers on kmean and this seperation it finds . how does it do it so well.

plt.show()
