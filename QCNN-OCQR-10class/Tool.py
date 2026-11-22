"""按既定随机顺序构造分类数据集。"""

import torchvision as tv
import torchvision.transforms as transforms

Path = '../CIFAR-10/Train/'
Download = True
import numpy as np
np.random.seed(0)
from torch.utils.data import DataLoader, Subset

# 沿用十分类采样顺序，训练集与测试集共用随机数流。
def sample_indices_per_class(labels, samples_per_class, category):
    
    labels = np.asarray(labels)
    if not 1 <= category <= 10:
        raise ValueError(f'category must be between 1 and 10, got {category}')
    chosen = []
    for class_id in range(category):
        class_indices = np.flatnonzero(labels == class_id)
        if len(class_indices) < samples_per_class:
            raise ValueError(f'Class {class_id} has only {len(class_indices)} samples')
        chosen.append(np.random.choice(class_indices, samples_per_class, replace=False))
    return np.concatenate(chosen).astype(int, copy=False)

def DataLoader_1280(batch_size=32, num_workers=4, input_size=32, Train_num=640, Test_num=100, category=10):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(input_size),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    full_train_dataset = tv.datasets.CIFAR10(
        root=Path,
        train=True,
        download=Download,
        transform=transform)

    labels = np.array(full_train_dataset.targets)

    all_selected_indices = sample_indices_per_class(labels, Train_num, category)

    train_dataset = Subset(full_train_dataset, all_selected_indices)
    print(len(train_dataset))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)
    full_test_dataset = tv.datasets.CIFAR10(
        root=Path,
        train=False,
        download=Download,
        transform=transform)

    labels = np.array(full_test_dataset.targets)

    all_selected_indices = sample_indices_per_class(labels, Test_num, category)
    test_dataset = Subset(full_test_dataset, all_selected_indices)
    print(len(test_dataset))

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=Test_num*category,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes
