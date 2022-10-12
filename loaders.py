import os.path as osp
import os
from PIL import Image
import numpy as np

import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MetaDataSet(Dataset):

    def __init__(self, root):
        data = []
        label = []
        lb = 0

        self.wnids = []
        folders = os.listdir(root)
        for folder in folders:
            folder_path = os.path.join(root, folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                data.append(file_path)
                label.append(lb)
            lb += 1


        self.data = data
        self.label = label

        self.tf = create_train_transform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.tf(Image.open(path).convert('RGB'))
        return image, label

def create_train_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf

def create_test_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return test_tf

class CategoriesSampler():
    #Taken from: https://github.com/yinboc/prototypical-network-pytorch/blob/master/samplers.py
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch