import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Fix for Tkinter error
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
import hdbscan
from sklearn.metrics import silhouette_score

# Define dataset path (Replace with your actual path)
lfw_dataset_path = r"D:\Software Engineering 3\Artificial Intelligence (Mr Fabrice)\Lab 4\Labeled Faces in the Wild (LFW) dataset\lfw"

# Function to load images from local LFW dataset
def load_lfw_images(dataset_path, img_size=(64, 64), max_images=1000):
    images = []
    labels = []
    person_names = os.listdir(dataset_path)  # Get subdirectories (each person)

    for person in person_names:
        person_path = os.path.join(dataset_path, person)
        if os.path.isdir(person_path):
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
                if img is not None:
                    img = cv2.resize(img, img_size)  # Resize image
                    images.append(img.flatten())  # Flatten image for PCA
                    labels.append(person)
                if len(images) >= max_images:  # Limit dataset size for efficiency
                    break

    return np.array(images), np.array(labels)

# Load LFW images
data, label_names = load_lfw_images(lfw_dataset_path)

# Normalize pixel values
data = data / 255.0

# Apply PCA for Dimensionality Reduction
pca = PCA(n_components=100)  # Retaining 100 principal components
data_pca = pca.fit_transform(data)

# Apply HDBSCAN Clustering with different distance metrics
for metric in ['euclidean', 'manhattan', 'cosine']:
    print(f"\nApplying HDBSCAN with {metric} metric:")

    if metric == "cosine":
        # Compute cosine distance matrix
        distance_matrix = cosine_distances(data_pca)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric="precomputed")
        labels = clusterer.fit_predict(distance_matrix)
    else:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric=metric)
        labels = clusterer.fit_predict(data_pca)

    # Visualize Clustering Results
    plt.figure(figsize=(8, 6))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis', s=5)
    plt.colorbar()
    plt.title(f"HDBSCAN Clustering with {metric} metric")
    plt.savefig(f"hdbscan_{metric}.png")  # Save the plot instead of displaying
    plt.close()

    # Identify noise points
    noise_points = np.sum(labels == -1)
    print(f"Number of noise points detected: {noise_points}")

    # Evaluate Clustering Quality
    silhouette = silhouette_score(data_pca, labels) if len(set(labels)) > 1 else "N/A"
    print(f"Silhouette Score: {silhouette}")

# Step 1: Introduce Noisy Data
def add_noise(image):
    noise = np.random.normal(loc=0, scale=25, size=image.shape)
    noisy_image = np.clip(image + noise, 0, 255)  # Ensure values remain valid
    return noisy_image.astype(np.uint8)

# Apply noise to a few images
noisy_images = [add_noise(img.reshape(64, 64)) for img in data[:5]]

# Save noisy images
for i, noisy_img in enumerate(noisy_images):
    cv2.imwrite(f"noisy_image_{i}.png", noisy_img)

# Step 2: Assign New Image to an Existing Cluster
def predict_cluster(image, model, pca_model):
    image = cv2.resize(image, (64, 64))
    image = image.flatten() / 255.0  # Normalize
    image_pca = pca_model.transform([image])
    label = model.fit_predict(image_pca)
    return label

# Example: Assign first noisy image to a cluster
new_label = predict_cluster(noisy_images[0], clusterer, pca)
print(f"The new image belongs to cluster: {new_label}")

# Step 3: Find the Most Representative Image for Each Cluster
def find_representative_images(data_pca, labels):
    unique_clusters = set(labels)
    representatives = {}
    for cluster in unique_clusters:
        if cluster == -1:
            continue  # Skip noise
        cluster_points = data_pca[labels == cluster]
        centroid = np.mean(cluster_points, axis=0)
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        representative_index = np.argmin(distances)
        representatives[cluster] = representative_index
    return representatives

# Get representative images
representatives = find_representative_images(data_pca, labels)

# Save representative images
for cluster, index in representatives.items():
    cv2.imwrite(f"cluster_{cluster}_representative.png", data[index].reshape(64, 64) * 255)

print("\nClustering completed. Check saved images for results.")
