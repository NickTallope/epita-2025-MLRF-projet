import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
import cv2

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

    print("Flattening images for training data...")
    train_flattened = flatten_images(train_data)
    print("Flattening images for test data...")
    test_flattened = flatten_images(test_data)

    np.save('../data/processed/train_hog_features.npy', train_hog_features)
    np.save('../data/processed/test_hog_features.npy', test_hog_features)
    np.save('../data/processed/train_sift_features.npy', train_sift_features)
    np.save('../data/processed/test_sift_features.npy', test_sift_features)
    np.save('../data/processed/train_flattened.npy', train_flattened)
    np.save('../data/processed/test_flattened.npy', test_flattened)

if __name__ == "__main__":
    main()
