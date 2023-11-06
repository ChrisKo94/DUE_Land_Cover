import torch
from torch.utils import data
from torchvision import datasets, transforms
import numpy as np
from skimage import io
import h5py

class custom_subset(data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset)  : The whole Dataset
        indices (sequence) : Indices in the whole set selected for subset
        labels  (sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices, labels):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = labels
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)

def iloader(path):
    image = np.asarray((io.imread(path)) / 255, dtype='float32')
    return image

def get_SVHN(root):
    input_size = 32
    num_classes = 10

    # NOTE: these are not correct mean and std for SVHN, but are commonly used
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = datasets.SVHN(
        root + "/SVHN", split="train", transform=transform, download=True
    )
    test_dataset = datasets.SVHN(
        root + "/SVHN", split="test", transform=transform, download=True
    )
    return input_size, num_classes, train_dataset, test_dataset


def get_CIFAR10(root):
    input_size = 32
    num_classes = 10

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    # Alternative
    # normalize = transforms.Normalize(
    #     (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    # )

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.CIFAR10(
        root + "/CIFAR10", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_dataset = datasets.CIFAR10(
        root + "/CIFAR10", train=False, transform=test_transform, download=False
    )

    return input_size, num_classes, train_dataset, test_dataset


def get_CIFAR100(root):
    input_size = 32
    num_classes = 100
    normalize = transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = datasets.CIFAR100(
        root + "/CIFAR100", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_dataset = datasets.CIFAR100(
        root + "/CIFAR100", train=False, transform=test_transform, download=False
    )

    return input_size, num_classes, train_dataset, test_dataset

def get_EUROSAT_ID(root):
    input_size = 64
    num_classes = 8

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    mapping_dict = {0:0, 1:1, 2:2, 3:3, 4:8, 5:4, 6:5, 7:9, 8:6, 9:7}

    data_full = datasets.DatasetFolder(root=root + "/EuroSAT/2750", loader=iloader, transform=transform, extensions='jpg')
    data_full.targets = list(map(mapping_dict.get, data_full.targets))

    id_targets = [0, 1, 2, 3, 4, 5, 6, 7]

    indices_id = {i for i, label in enumerate(data_full.targets) if label in id_targets}
    data_id = data.Subset(data_full, indices_id)

    np.random.seed(42)
    indices_train = np.random.choice(np.array(list(indices_id)),
                                     np.floor(len(indices_id)*0.7).astype(int),
                                     replace=False)
    indices_train = np.sort(indices_train)
    indices_test = {i for i in indices_id if i not in set(list(indices_train))}
    indices_test = np.array(list(indices_test))
    indices_test = np.sort(indices_test)

    labels_train = np.array(data_full.targets)[indices_train]
    labels_test = np.array(data_full.targets)[indices_test]

    train_id = custom_subset(data_full, indices_train, labels_train)
    test_id = custom_subset(data_full, indices_test, labels_test)
    return input_size, num_classes, train_id, test_id

def get_EUROSAT_OOD(root):
    input_size = 64
    num_classes = 8

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    mapping_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 8, 5: 4, 6: 5, 7: 9, 8: 6, 9: 7}

    data_full = datasets.DatasetFolder(root=root + "/EuroSAT/2750", loader=iloader, transform=transform, extensions='jpg')
    data_full.targets = list(map(mapping_dict.get, data_full.targets))


    id_targets = [8, 9]

    indices_id = {i for i, label in enumerate(data_full.targets) if label in id_targets}
    data_id = data.Subset(data_full, indices_id)

    np.random.seed(42)
    indices_train = np.random.choice(np.array(list(indices_id)),
                                     np.floor(len(indices_id)*0.7).astype(int),
                                     replace=False)
    indices_train = np.sort(indices_train)
    indices_test = {i for i in indices_id if i not in set(list(indices_train))}
    indices_test = np.array(list(indices_test))

    labels_train = np.array(data_full.targets)[indices_train]
    labels_test = np.array(data_full.targets)[indices_test]

    train_id = custom_subset(data_full, indices_train, labels_train)
    test_id = custom_subset(data_full, indices_test, labels_test)
    # Since get_ood_metrics uses test sets, we shift the train ood set to the 4th position (otherwise too small)
    return input_size, num_classes, test_id, train_id

#
#Todo: Check if labels are one-hot, transform

def get_LCZ42_Veg(root):
    input_size = 32
    num_classes = 7
    normalize = transforms.Normalize((0.1266, 0.1017, 0.0776), (0.0222, 0.2509, 0.0352))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    data_full_h5 = h5py.File(root+"/LCZ42/training.h5")
    labels = np.array(data_full_h5["label"])
    images = np.array(data_full_h5["sen2"][:, :, :, :3]).transpose(0,3,1,2).astype('float32')

    mapping_dict = {0: 7, 1: 8, 2: 9, 3: 10, 4: 11, 5: 12, 6: 13, 7: 14, 8: 15, 9: 16,
                    10: 0, 11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6}
    labels = np.argmax(labels,1)
    labels = np.array(list(map(mapping_dict.get, labels)))
    data_full = data.TensorDataset(torch.from_numpy(images),torch.from_numpy(labels))

    indices_id = np.where(labels < 7)[0]
    np.random.seed(42)
    indices_train = np.random.choice(indices_id,
                                     np.floor(len(indices_id)*0.7).astype(int),
                                     replace=False)
    indices_train = np.sort(indices_train)
    indices_test = set(list(indices_id)) - set(list(indices_train))
    indices_test = np.array(list(indices_test))

    labels_train = labels[indices_train]
    labels_test = labels[indices_test]

    train_id = custom_subset(data_full, indices_train, labels_train)
    test_id = custom_subset(data_full, indices_test, labels_test)
    # Since get_ood_metrics uses test sets, we shift the train ood set to the 4th position (otherwise too small)
    return input_size, num_classes, train_id, test_id

def get_LCZ42_Build(root):
    input_size = 32
    num_classes = 10
    normalize = transforms.Normalize((0.1266, 0.1017, 0.0776), (0.0222, 0.2509, 0.0352))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    data_full_h5 = h5py.File(root+"/LCZ42/training.h5")
    labels = np.array(data_full_h5["label"])
    images = np.array(data_full_h5["sen2"][:, :, :, :3]).transpose(0,3,1,2).astype('float32')

    labels = np.argmax(labels,1)
    data_full = data.TensorDataset(torch.from_numpy(images),torch.from_numpy(labels))

    indices_id = np.where(labels < 10)[0]
    np.random.seed(42)
    indices_train = np.random.choice(indices_id,
                                     np.floor(len(indices_id)*0.7).astype(int),
                                     replace=False)
    indices_train = np.sort(indices_train)
    indices_test = set(list(indices_id)) - set(list(indices_train))
    indices_test = np.array(list(indices_test))

    labels_train = labels[indices_train]
    labels_test = labels[indices_test]

    train_id = custom_subset(data_full, indices_train, labels_train)
    test_id = custom_subset(data_full, indices_test, labels_test)
    # Since get_ood_metrics uses test sets, we shift the train ood set to the 4th position (otherwise too small)
    return input_size, num_classes, train_id, test_id

all_datasets = {
    "SVHN": get_SVHN,
    "EUROSAT_ID": get_EUROSAT_ID,
    "EUROSAT_OOD": get_EUROSAT_OOD,
    "LCZ42_Veg": get_LCZ42_Veg,
    "LCZ42_Build": get_LCZ42_Build,
    "CIFAR10": get_CIFAR10,
    "CIFAR100": get_CIFAR100,
}


def get_dataset(dataset, root="./"):
    return all_datasets[dataset](root)


def get_dataloaders(dataset, train_batch_size=128, root="./"):
    ds = all_datasets[dataset](root)
    input_size, num_classes, train_dataset, test_dataset = ds

    kwargs = {"num_workers": 4, "pin_memory": True}

    train_loader = data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs
    )

    test_loader = data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False, **kwargs
    )

    return train_loader, test_loader, input_size, num_classes

'''_,_, ds_train, ds_test = get_dataset("EUROSAT_ID", root="D:/Dateien/")
train_loader = data.DataLoader(ds_train, batch_size=32, shuffle=True)
test_loader = data.DataLoader(ds_test, batch_size=32, shuffle=False)

x,y=next(iter(train_loader))
x,y=next(iter(test_loader))

x.astype(int)'''
