import os
import struct
import numpy as np


def load_images(file_path):
    """Load MNIST images from the raw idx file."""
    with open(file_path, 'rb') as f:
        # read header information
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # load the rest of the data into a numpy array and reshape
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
    return images


def load_labels(file_path):
    """Load MNIST labels from the raw idx file."""
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_dataset(data_dir):
    """
    Load training and test datasets from a given directory.

    Expected files:
      - train-images-idx3-ubyte
      - train-labels-idx1-ubyte
      - t10k-images-idx3-ubyte
      - t10k-labels-idx1-ubyte
    """
    paths = {
        "train_images": os.path.join(data_dir, "train-images-idx3-ubyte"),
        "train_labels": os.path.join(data_dir, "train-labels-idx1-ubyte"),
        "test_images": os.path.join(data_dir, "t10k-images-idx3-ubyte"),
        "test_labels": os.path.join(data_dir, "t10k-labels-idx1-ubyte")
    }

    train_images = load_images(paths["train_images"])
    train_labels = load_labels(paths["train_labels"])
    test_images = load_images(paths["test_images"])
    test_labels = load_labels(paths["test_labels"])

    # Optionally, normalize the images to [0,1] and add a channel dimension.
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    train_images = np.expand_dims(train_images, axis=1)  # shape: (N, 1, 28, 28)
    test_images = np.expand_dims(test_images, axis=1)

    return (train_images, train_labels), (test_images, test_labels)


# Test the loader when running this script directly
if __name__ == '__main__':
    data_dir = "/Users/kennyyu/Desktop/projects/AI/MNIST/mnist/MNIST/raw"
    (train_images, train_labels), (test_images, test_labels) = load_dataset(data_dir)
    print("Train images shape:", train_images.shape)
    print("Train labels shape:", train_labels.shape)
    print("Test images shape:", test_images.shape)
    print("Test labels shape:", test_labels.shape)
