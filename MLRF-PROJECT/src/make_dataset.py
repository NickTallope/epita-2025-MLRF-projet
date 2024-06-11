import numpy as np
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10(data_dir):
    train_data = []
    train_labels = []
    for i in range(1, 6):
        batch = unpickle(f'{data_dir}/data_batch_{i}')
        train_data.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])
    
    train_data = np.concatenate(train_data)
    train_data = train_data.reshape((50000, 3, 32, 32)).transpose(0, 2, 3, 1)
    train_labels = np.array(train_labels)

    test_batch = unpickle(f'{data_dir}/test_batch')
    test_data = test_batch[b'data']
    test_labels = test_batch[b'labels']
    test_data = test_data.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = np.array(test_labels)

    return (train_data, train_labels), (test_data, test_labels)

def main():
    data_dir = '../data/raw/cifar-10-batches-py'
    (train_data, train_labels), (test_data, test_labels) = load_cifar10(data_dir)
    
    # Save processed data
    np.save('../data/processed/train_data.npy', train_data)
    np.save('../data/processed/train_labels.npy', train_labels)
    np.save('../data/processed/test_data.npy', test_data)
    np.save('../data/processed/test_labels.npy', test_labels)

if __name__ == "__main__":
    main()
