import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
import cv2
from sklearn.decomposition import PCA

def apply_pca(images, n_components=100):
    flat_images = images.reshape(images.shape[0], -1)
    pca = PCA(n_components=n_components)
    reduced_images = pca.fit_transform(flat_images)
    return reduced_images

def extract_hog_features(images):
    hog_features = []
    for image in images:
        gray_image = rgb2gray(image.reshape(32, 32, 3))
        features = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        hog_features.append(features)
    return np.array(hog_features)

def extract_sift_features_single_image(image, max_features=500):
    sift = cv2.SIFT_create()
    gray_image = cv2.cvtColor(image.reshape(32, 32, 3).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    kp, des = sift.detectAndCompute(gray_image, None)
    if des is not None:
        if des.shape[0] > max_features:
            des = des[:max_features, :]
        else:
            des = np.pad(des, ((0, max_features - des.shape[0]), (0, 0)), mode='constant')
        return des.flatten()
    else:
        return np.zeros(max_features * 128)

def extract_sift_features(images, max_features=500, batch_size=1000):
    sift_features = []
    num_images = images.shape[0]
    for start in range(0, num_images, batch_size):
        end = min(start + batch_size, num_images)
        batch_features = []
        for image in images[start:end]:
            batch_features.append(extract_sift_features_single_image(image, max_features))
        sift_features.append(np.array(batch_features))
    return np.concatenate(sift_features, axis=0)

def main():
    train_data = np.load('../data/processed/train_data.npy')
    test_data = np.load('../data/processed/test_data.npy')

    # Apply PCA for feature reduction
    print("Applying PCA for training data...")
    train_reduced = apply_pca(train_data)
    print("Applying PCA for test data...")
    test_reduced = apply_pca(test_data)

    # Extract features from PCA-reduced data
    print("Extracting HOG features for training data...")
    train_hog_features = extract_hog_features(train_reduced)
    print("Extracting HOG features for test data...")
    test_hog_features = extract_hog_features(test_reduced)

    print("Extracting SIFT features for training data...")
    train_sift_features = extract_sift_features(train_reduced)
    print("Extracting SIFT features for test data...")
    test_sift_features = extract_sift_features(test_reduced)

    # Save processed data
    np.save('../data/processed/train_hog_features.npy', train_hog_features)
    np.save('../data/processed/test_hog_features.npy', test_hog_features)
    np.save('../data/processed/train_sift_features.npy', train_sift_features)
    np.save('../data/processed/test_sift_features.npy', test_sift_features)

if __name__ == "__main__":
    main()
