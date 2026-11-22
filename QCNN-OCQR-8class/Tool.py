"""按既定随机顺序构造分类数据集。"""

import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

Path='../CIFAR-10/Train/'
Download=True

CIFAR_CLASSES=('airplane','automobile','bird','cat','deer','dog',
               'frog','horse','ship','truck')

# 沿用十分类采样顺序，训练集与测试集共用随机数流。
def frozen_tenclass_subset_indices(train_targets, test_targets,
                                   train_per_class=640, test_per_class=100,
                                   num_classes=8, seed=0):
    if not 2 <= num_classes <= 10:
        raise ValueError('num_classes must be in [2,10]')
    tr_labels=np.asarray(train_targets)
    te_labels=np.asarray(test_targets)
    rng=np.random.RandomState(seed)

    def draw_all(labels, amount):
        chosen=[]
        for cls in range(10):
            available=np.flatnonzero(labels==cls)
            if len(available)<amount:
                raise ValueError(f'Class {cls}: {len(available)} images, need {amount}')
            selection=rng.choice(available, amount, replace=False)
            if cls<num_classes:
                chosen.append(selection)
        return np.concatenate(chosen).astype(np.int64,copy=False)
    return draw_all(tr_labels,train_per_class),draw_all(te_labels,test_per_class)

def DataLoader_1280(batch_size=128,num_workers=8,input_size=10,
                    Train_num=640,Test_num=100,category=8):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(input_size),
        transforms.Normalize((.5,.5,.5),(.5,.5,.5)),
    ])
    full_train=datasets.CIFAR10(root=Path, train=True, download=Download, transform=transform)
    full_test=datasets.CIFAR10(root=Path, train=False, download=Download, transform=transform)
    train_ids,test_ids=frozen_tenclass_subset_indices(
        full_train.targets,full_test.targets,
        train_per_class=Train_num,test_per_class=Test_num,
        num_classes=category,seed=0,
    )
    train_loader=DataLoader(Subset(full_train,train_ids),batch_size=batch_size,
                            shuffle=True,num_workers=num_workers,pin_memory=True)
    test_loader=DataLoader(Subset(full_test,test_ids),batch_size=Test_num*category,
                           shuffle=True,num_workers=num_workers,pin_memory=True)
    return train_loader,test_loader,CIFAR_CLASSES[:category]
