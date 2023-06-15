import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CIFAR10Dataset(Dataset):
    def __init__(self, data_file, mode='train'):
        self.data_file = data_file
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        self.data, self.labels = self.load_single_batch()

    def load_single_batch(self):
        with open(self.data_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')

        data = batch['data']
        labels = np.array(batch['labels'])

        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index].reshape(3, 32, 32).astype(np.float32)
        label = self.labels[index]
        label = np.eye(10)[label]

        sample = torch.from_numpy(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample, label

# Example usage
if __name__ == "__main__":
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    # Set the path to one of the new CIFAR-10 batch files
    data_file = './cifar-10-batches-py/repack1/new_random_data_batch_1'

    # Create a dataset
    #transform = transforms.Compose([
    #    transforms.ToPILImage(),
    #    transforms.RandomHorizontalFlip(),
    #    transforms.RandomCrop(32, padding=4),
    #    transforms.ToTensor()
    #])
    dataset = CIFAR10Dataset(data_file)

    # Create a data loader
    batch_size = 32 
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Iterate over the data loader
    for i, (images, labels) in enumerate(data_loader):
        #images = batch['data']
        #labels = batch['label']

        # Process the images and labels
        # e.g., pass them through your neural network
        print(f"Batch {i + 1}:")
        #print(f"Images shape: {images.shape}")  # (batch_size, 3, 32, 32)
        #print(f"Labels shape: {labels.shape}")  # (batch_size,)
        print(labels)
