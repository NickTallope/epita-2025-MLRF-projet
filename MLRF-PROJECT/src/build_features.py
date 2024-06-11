import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
import cv2
from sklearn.decomposition import IncrementalPCA

def extract_hog_features(images):
    hog_features = []
    for image in images:
        gray_image = rgb2gray(image)
        features = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        hog_features.append(features)
    return np.array(hog_features)

def extract_sift_features_single_image(image, max_features=500):
    sift = cv2.SIFT_create()
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kp, des = sift.detectAndCompute(gray_image, None)
    if des is not None:
        if des.shape[0] > max_features:
            des = des[:max_features, :]
        else:
            des = np.pad(des, ((0, max_features - des.shape[0]), (0, 0)), mode='constant')
        return des.flatten()
    else:
        return np.zeros(max_features * 128)

def extract_sift_features(images, max_features=500):
    sift_features = []
    for image in images:
        sift_features.append(extract_sift_features_single_image(image, max_features))
    return np.array(sift_features)

def apply_incremental_pca(features, n_components=100, batch_size=200):
    ipca = IncrementalPCA(n_components=n_components)
    num_samples = features.shape[0]

    # Fit in batches
    for i in range(0, num_samples, batch_size):
        batch = features[i:i + batch_size]
        ipca.partial_fit(batch)

    # Transform in batches
    reduced_features = []
    for i in range(0, num_samples, batch_size):
        batch = features[i:i + batch_size]
        reduced_features.append(ipca.transform(batch))

    return np.vstack(reduced_features)

def flatten_images(images):
    return images.reshape(images.shape[0], -1)

def main():
    train_data = np.load('../data/processed/train_data.npy')
    test_data = np.load('../data/processed/test_data.npy')

    print("Extracting HOG features for training data...")
    train_hog_features = extract_hog_features(train_data)
    print("Extracting HOG features for test data...")
    test_hog_features = extract_hog_features(test_data)

    print("Extracting SIFT features for training data...")
    train_sift_features = extract_sift_features(train_data)
    print("Extracting SIFT features for test data...")
    test_sift_features = extract_sift_features(test_data)

    print("Applying Incremental PCA to HOG features for training data...")
    train_hog_pca = apply_incremental_pca(train_hog_features)
    print("Applying Incremental PCA to HOG features for test data...")
    test_hog_pca = apply_incremental_pca(test_hog_features)

    print("Applying Incremental PCA to SIFT features for training data...")
    train_sift_pca = apply_incremental_pca(train_sift_features)
    print("Applying Incremental PCA to SIFT features for test data...")
    test_sift_pca = apply_incremental_pca(test_sift_features)

    np.save('../data/processed/train_hog_features.npy', train_hog_features)
    np.save('../data/processed/test_hog_features. npy', test_hog_features)
    np.save('../data/processed/train_hog_pca.npy', train_hog_pca)
    np.save('../data/processed/test_hog_pca.npy', test_hog_pca)
    np.save('../data/processed/train_sift_features.npy', train_sift_features)
    np.save('../data/processed/test_sift_features.npy', test_sift_features)
    np.save('../data/processed/train_sift_pca.npy', train_sift_pca)
    np.save('../data/processed/test_sift_pca.npy', test_sift_pca)

if __name__ == "__main__":
    main()