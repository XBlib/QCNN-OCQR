"""训练增强、验证集划分与模型评估。"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision as tv
from torchvision import transforms

def original_eval_transform(image_size=10):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
    ])

def train_transform(image_size=10, augment="flip"):
    pre = []
    if augment in ("flip", "flip-jitter"):
        pre.append(transforms.RandomHorizontalFlip(p=0.5))
    elif augment != "none":
        raise ValueError("augment must be none / flip / flip-jitter")
    if augment == "flip-jitter":
        pre.append(transforms.ColorJitter(brightness=.08, contrast=.08,
                                          saturation=.05, hue=.01))
    return transforms.Compose(pre + [
        transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
    ])

# 验证样本只从未参与训练的索引中抽取。
def holdout_indices(targets, train_indices, per_class=100, seed=314159, num_classes=3):
    
    targets=np.asarray(targets)
    if targets.ndim != 1:
        raise ValueError("CIFAR10 target labels must be 1 dimensional")
    train_indices=np.asarray(train_indices, dtype=int)
    if len(np.unique(train_indices)) != len(train_indices):
        raise ValueError("Duplicate original train index")
    rng=np.random.default_rng(seed)
    unused=np.ones(len(targets), dtype=bool)
    unused[train_indices]=False
    chosen=[]
    if not 1 <= num_classes <= 10:
        raise ValueError("num_classes must be between 1 and 10")
    for klass in range(num_classes):
        available=np.flatnonzero((targets == klass) & unused)
        if len(available)<per_class:
            raise ValueError(f"Class {klass}: only {len(available)} unused training images")
        chosen.extend(rng.choice(available, per_class, replace=False).tolist())
    chosen=np.asarray(chosen,dtype=int)
    assert not (set(chosen.tolist()) & set(train_indices.tolist()))
    return chosen

def build_validation_loader(data_root, download, train_indices,
                            num_workers=8, per_class=100, seed=314159,
                            image_size=10, batch_size=300, num_classes=3):
    dataset=tv.datasets.CIFAR10(
        root=data_root, train=True, download=download,
        transform=original_eval_transform(image_size)
    )
    selected=holdout_indices(dataset.targets, train_indices,
                            per_class=per_class, seed=seed, num_classes=num_classes)
    val_set=Subset(dataset, selected.tolist())
    loader=DataLoader(val_set,batch_size=batch_size,shuffle=False,
                      num_workers=num_workers,pin_memory=True)
    return loader, selected

@torch.no_grad()
def evaluate(net, loader, device):
    

    import torch.nn.functional as F
    net.eval()
    total, correct, loss_total = 0,0,0.0
    for x,y in loader:
        x=x.to(device);y=y.to(device)
        logits=net(x)
        loss_total+=F.cross_entropy(logits,y,reduction='sum').item()
        correct+=(logits.argmax(dim=1)==y).sum().item()
        total+=y.numel()
    if not total: raise ValueError('Empty eval loader')
    return {'count':int(total), 'acc_percent':100.0*correct/total,
            'loss':loss_total/total}
