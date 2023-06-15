import os
import tarfile
import urllib.request

# Set the URL and filename for the dataset
url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
filename = "cifar-10-python.tar.gz"

# Download the dataset if it does not exist
if not os.path.exists(filename):
    print("Downloading CIFAR-10 dataset...")
    urllib.request.urlretrieve(url, filename)

# Extract the dataset if it has not been extracted
if not os.path.exists("cifar-10-batches-py"):
    print("Extracting CIFAR-10 dataset...")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall()

print("CIFAR-10 dataset is ready.")
