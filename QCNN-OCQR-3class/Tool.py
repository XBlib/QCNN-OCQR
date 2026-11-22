"""按既定随机顺序构造分类数据集。"""

import numpy as np

Path='../CIFAR-10/Train/'
Download=True

# 先按六分类顺序抽样，再保留目标类别。
def legacy_sixclass_subset_indices(train_targets, test_targets,
                                   train_per_class=640, test_per_class=100,
                                   num_classes=3, seed=0):
    

    if not 2 <= num_classes <= 6:
        raise ValueError('num_classes must be in [2,6] for legacy six-class sampler')
    train_targets=np.asarray(train_targets)
    test_targets=np.asarray(test_targets)
    rng=np.random.RandomState(seed)
    def draw_all(targets, amount):
        chunks=[]
        for k in range(6):
            available=np.flatnonzero(targets==k)
            if len(available)<amount:
                raise ValueError(f'Class {k} has only {len(available)} examples; need {amount}')
            drawn=rng.choice(available,amount,replace=False)
            if k<num_classes:
                chunks.append(drawn)
        return np.concatenate(chunks).astype(int,copy=False)
    selected_train=draw_all(train_targets,train_per_class)
    selected_test=draw_all(test_targets,test_per_class)
    return selected_train,selected_test

def DataLoader_1280(batch_size=128,num_workers=8,input_size=10,
                    Train_num=640,Test_num=100,category=3):

    from torchvision import datasets, transforms
    from torch.utils.data import Subset, DataLoader
    tfm=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(input_size),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
    full_train=datasets.CIFAR10(root=Path,train=True,download=Download,transform=tfm)
    full_test=datasets.CIFAR10(root=Path,train=False,download=Download,transform=tfm)
    train_ids,test_ids=legacy_sixclass_subset_indices(
        full_train.targets,full_test.targets,train_per_class=Train_num,
        test_per_class=Test_num,num_classes=category,seed=0)
    train_loader=DataLoader(Subset(full_train,train_ids),batch_size=batch_size,
                            shuffle=True,num_workers=num_workers,pin_memory=True)
    test_loader=DataLoader(Subset(full_test,test_ids),batch_size=Test_num*category,
                           shuffle=True,num_workers=num_workers,pin_memory=True)
    names=('airplane','automobile','bird','cat','deer','dog',
           'frog','horse','ship','truck')
    return train_loader,test_loader,names[:category]
