import os
import pickle
import numpy as np

def load_cifar10_batch(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data[b'data'], np.array(data[b'labels'])

def save_cifar10_batch(data, labels, file):
    with open(file, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

def generate_random_sizes(num_samples, num_batches):
    sizes = np.random.randint(1, num_samples - num_batches, size=num_batches - 1)
    sizes.sort()
    sizes = np.append(sizes, num_samples - num_batches)
    sizes = np.diff(np.concatenate(([0], sizes)))
    np.random.shuffle(sizes)
    return sizes

data_dir = 'cifar-10-batches-py'

# Load the 5 original batches
data_batches = []
label_batches = []
for i in range(1, 6):
    file = os.path.join(data_dir, f'data_batch_{i}')
    data, labels = load_cifar10_batch(file)
    data_batches.append(data)
    label_batches.append(labels)

# Combine the batches
combined_data = np.concatenate(data_batches)
combined_labels = np.concatenate(label_batches)

# Split the combined data into 6 new random-sized batches
num_batches = 6
random_sizes = generate_random_sizes(len(combined_data), num_batches)

start = 0
for i, size in enumerate(random_sizes):
    end = start + size
    data = combined_data[start:end]
    labels = combined_labels[start:end]

    save_file = os.path.join(data_dir, f'new_random_data_batch_{i+1}')
    save_cifar10_batch(data, labels, save_file)

    # Print the total number and the number of each class in the new batch
    print(f"Batch {i+1}:")
    print(f"  Total number of samples: {size}")
    for class_id in range(10):
        class_count = np.sum(labels == class_id)
        print(f"  Number of samples in class {class_id}: {class_count}")

    start = end

print("The combined dataset has been split into 6 new random-sized batches.")
