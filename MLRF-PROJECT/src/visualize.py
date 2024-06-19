import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_latent_space(features, labels):
    pca = PCA(n_components=2)
    transformed_features = pca.fit_transform(features)
    
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(transformed_features[:, 0], transformed_features[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, ticks=range(10))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of CIFAR-10 Features')
    plt.show()

def main():
    features = np.load('../data/processed/train_flattened.npy')
    labels = np.load('../data/processed/train_labels.npy')
    visualize_latent_space(features, labels)

if __name__ == "__main__":
    main()
