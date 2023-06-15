import numpy as np
import os
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def print_batch_info(batch_file):
    batch_data = unpickle(batch_file)
    labels = batch_data[b'labels']
    total_images = len(labels)
    class_counts = np.zeros(10, dtype=int)

    for label in labels:
        class_counts[label] += 1

    print(f'Total images in {os.path.basename(batch_file)}: {total_images}')
    for i in range(10):
        print(f'Class {i}: {class_counts[i]}')

data_dir = 'cifar-10-batches-py'  # replace with your dataset directory
batch_files = [os.path.join(data_dir, f'data_batch_{i}') for i in range(1, 6)]

for batch_file in batch_files:
    print_batch_info(batch_file)
    print()
