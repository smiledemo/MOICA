import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
import os


class OfficeHomeDataset(Dataset):
    def __init__(self, root_dir, domain, transform=None):
        self.root_dir = root_dir
        self.domain = domain
        self.transform = transform
        self.image_paths, self.labels = self._load_data()

    def _load_data(self):
        image_paths = []
        labels = []
        domain_path = os.path.join(self.root_dir, self.domain)
        for label, class_dir in enumerate(os.listdir(domain_path)):
            class_dir_path = os.path.join(domain_path, class_dir)
            for img_name in os.listdir(class_dir_path):
                image_paths.append(os.path.join(class_dir_path, img_name))
                labels.append(label)
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class Office31Dataset(Dataset):
    def __init__(self, root_dir, domain, transform=None):
        self.root_dir = root_dir
        self.domain = domain
        self.transform = transform
        self.image_paths, self.labels = self._load_data()

    def _load_data(self):
        image_paths = []
        labels = []
        domain_path = os.path.join(self.root_dir, self.domain)
        for label, class_dir in enumerate(os.listdir(domain_path)):
            class_dir_path = os.path.join(domain_path, class_dir)
            for img_name in os.listdir(class_dir_path):
                image_paths.append(os.path.join(class_dir_path, img_name))
                labels.append(label)
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class DigitDataset(Dataset):
    def __init__(self, root_dir, dataset_name, transform=None):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.transform = transform
        self.data, self.labels = self._load_data()

    def _load_data(self):
        if self.dataset_name == 'MNIST':
            dataset = datasets.MNIST(self.root_dir, train=True, download=True)
        elif self.dataset_name == 'SVHN':
            dataset = datasets.SVHN(self.root_dir, split='train', download=True)
        elif self.dataset_name == 'USPS':
            dataset = datasets.USPS(self.root_dir, train=True, download=True)
        else:
            raise ValueError('Unknown dataset: {}'.format(self.dataset_name))

        data = dataset.data
        labels = dataset.targets
        if self.dataset_name == 'SVHN':
            data = data.transpose((0, 2, 3, 1))
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)
        return image, label


def get_dataloaders(dataset_name, domain=None, batch_size=32, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    if dataset_name == 'OfficeHome':
        train_dataset = OfficeHomeDataset(root_dir='path/to/OfficeHome', domain=domain, transform=transform)
        val_dataset = OfficeHomeDataset(root_dir='path/to/OfficeHome', domain=domain, transform=transform)
    elif dataset_name == 'Office31':
        train_dataset = Office31Dataset(root_dir='path/to/Office31', domain=domain, transform=transform)
        val_dataset = Office31Dataset(root_dir='path/to/Office31', domain=domain, transform=transform)
    elif dataset_name in ['MNIST', 'SVHN', 'USPS']:
        train_dataset = DigitDataset(root_dir='path/to/digit', dataset_name=dataset_name, transform=transform)
        val_dataset = DigitDataset(root_dir='path/to/digit', dataset_name=dataset_name, transform=transform)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_name))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def get_test_loader(dataset_name, domain=None, batch_size=32, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    if dataset_name == 'OfficeHome':
        test_dataset = OfficeHomeDataset(root_dir='path/to/OfficeHome', domain=domain, transform=transform)
    elif dataset_name == 'Office31':
        test_dataset = Office31Dataset(root_dir='path/to/Office31', domain=domain, transform=transform)
    elif dataset_name in ['MNIST', 'SVHN', 'USPS']:
        test_dataset = DigitDataset(root_dir='path/to/digit', dataset_name=dataset_name, transform=transform)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_name))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader
