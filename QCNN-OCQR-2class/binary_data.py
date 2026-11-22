
"""构建飞机与汽车二分类数据集。"""

import numpy as np
import torchvision as tv
from torch.utils.data import DataLoader, Subset
from generalization import original_eval_transform, train_transform

CLASS_IDS = (0, 1)
CLASS_NAMES = ('airplane', 'automobile')

def legacy_aligned_binary_indices(train_targets, test_targets, train_per_class=640,
                                  test_per_class=100, seed=0):
    # 先抽取六类，再保留前两类，以复现原采样顺序。
    rng = np.random.RandomState(seed)
    train_targets = np.asarray(train_targets)
    test_targets = np.asarray(test_targets)
    train_chunks=[]
    for k in range(6):
        possibilities=np.flatnonzero(train_targets == k)
        if len(possibilities) < train_per_class:
            raise ValueError(f'Not enough class {k} train examples')
        drawn=rng.choice(possibilities, train_per_class, replace=False)
        if k in CLASS_IDS: train_chunks.append(drawn)
    test_chunks=[]
    for k in range(6):
        possibilities=np.flatnonzero(test_targets == k)
        if len(possibilities) < test_per_class:
            raise ValueError(f'Not enough class {k} test examples')
        drawn=rng.choice(possibilities, test_per_class, replace=False)
        if k in CLASS_IDS: test_chunks.append(drawn)
    return np.concatenate(train_chunks), np.concatenate(test_chunks)

def build_binary_loaders(data_root, download, batch_size=128, num_workers=8,
                         image_size=10, train_per_class=640, test_per_class=100,
                         augment='flip'):
    train_full=tv.datasets.CIFAR10(root=data_root,train=True,download=download,
                                   transform=train_transform(image_size,augment))
    test_full=tv.datasets.CIFAR10(root=data_root,train=False,download=download,
                                  transform=original_eval_transform(image_size))
    train_idx,test_idx=legacy_aligned_binary_indices(
        train_full.targets,test_full.targets,train_per_class,test_per_class)
    for targets,idx in ((train_full.targets,train_idx),(test_full.targets,test_idx)):
        labels=np.asarray(targets)[idx]
        if set(labels.tolist()) != set(CLASS_IDS):
            raise ValueError('Binary subset classes were not selected correctly')
    tr=DataLoader(Subset(train_full,train_idx.tolist()),batch_size=batch_size,
                  shuffle=True,num_workers=num_workers,pin_memory=True)
    te=DataLoader(Subset(test_full,test_idx.tolist()),batch_size=test_per_class*2,
                  shuffle=False,num_workers=num_workers,pin_memory=True)
    return tr,te,train_idx,test_idx
