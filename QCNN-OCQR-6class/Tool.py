"""按既定随机顺序构造分类数据集。"""

import numpy as np
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

Path = '../CIFAR-10/Train/'
Download = True
np.random.seed(0)

# 训练集与测试集按同一随机数流依次抽取六类。
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
    class_indices = {i: np.where(labels == i)[0] for i in range(category)}
    selected_indices = {i: np.random.choice(class_indices[i], Train_num, replace=False) for i in [0, 1, 2, 3, 4, 5]}
    all_selected_indices = np.concatenate(list(selected_indices.values()))
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
    class_indices = {i: np.where(labels == i)[0] for i in range(category)}
    selected_indices = {i: np.random.choice(class_indices[i], Test_num, replace=False) for i in [0, 1, 2, 3, 4, 5]}
    all_selected_indices = np.concatenate(list(selected_indices.values()))
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
