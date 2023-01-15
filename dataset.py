import glob
import os

from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset

class IDRiDataset(Dataset):
    def __init__(self, inputs_root, labels_root, transform):
    # def __init__(self, inputs_root, labels_root):
        # print(os.getcwd())
        # print(f'{inputs_root}/*.jpg')
        self.files = sorted(glob.glob(f'{inputs_root}/*.jpg'))
        # self.files = sorted(glob.glob("../data/A. Segmentation/1. Original Images/a. Training Set/*.jpg"))
        # print('len of Dataset: ' + str(len(self.files)))
        self.imgs = sorted(glob.glob(f'{inputs_root}/*.jpg'))
        # self.imgs = sorted(glob.glob("../data/A. Segmentation/1. Original Images/a. Training Set/*.jpg"))
        # print('len of imgs: ' + str(len(self.imgs)))
        self.files_using = sorted(glob.glob(f'{labels_root}/*.tif'))
        # self.files_using = sorted(glob.glob("../data/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/4. Soft Exudates/*.tif"))
        self.transform = transform


    def __getitem__(self, index):
        inputs = plt.imread(self.files[index % len(self.files)])
        # inputs = plt.imread(self.imgs[index % len(self.imgs)])
        labels = plt.imread(self.files_using[index % len(self.files_using)])
        # inputs = Image.open(self.files[index % len(self.files)]).convert('RGB')
        # labels = Image.open(self.files_using[index % len(self.files_using)]).convert('RGB')
        if self.transform is not None:
            inputs = self.transform(Image.fromarray(inputs))
            labels = self.transform(Image.fromarray(labels))
        return inputs, labels

    def __len__(self):
        return len(self.files)
        # return len(self.imgs)

class DDRnonDataset(Dataset):
    def __init__(self, inputs_root, transform):
        self.files = sorted(glob.glob(f'{inputs_root}/*.jpg'))
        self.transform = transform

    def __getitem__(self, index):
        inputs = plt.imread(self.files[index % len(self.files)])
        if self.transform is not None:
            inputs = self.transform(Image.fromarray(inputs))
        return inputs

    def __len__(self):
        return len(self.files)

#DDR数据集
class DDRDataset(Dataset):
    def __init__(self, inputs_root, labels_root, transform):
    # def __init__(self, inputs_root, labels_root):
        # print(os.getcwd())
        # print(f'{inputs_root}/*.jpg')
        self.files = sorted(glob.glob(f'{inputs_root}/*.jpg'))
        # self.files = sorted(glob.glob("../data/A. Segmentation/1. Original Images/a. Training Set/*.jpg"))
        # print('len of Dataset: ' + str(len(self.files)))
        self.files_using = sorted(glob.glob(f'{labels_root}/*.tif'))
        # self.files_using = sorted(glob.glob("../data/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/4. Soft Exudates/*.tif"))
        self.transform = transform




    def __getitem__(self, index):
        inputs = plt.imread(self.files[index % len(self.files)])
        labels = plt.imread(self.files_using[index % len(self.files_using)])
        if self.transform is not None:
            inputs = self.transform(Image.fromarray(inputs))
            labels = self.transform(Image.fromarray(labels))
        return inputs, labels

    def __len__(self):
        return len(self.files)

#DDR+血管信息
class DDRvDataset(Dataset):
    def __init__(self, inputs_root, vessels_root, labels_root, transform):
        self.files = sorted(glob.glob(f'{inputs_root}/*.jpg'))
        self.vessels = sorted(glob.glob(f'{vessels_root}/*.tif'))
        self.files_using = sorted(glob.glob(f'{labels_root}/*.tif'))
        self.transform = transform




    def __getitem__(self, index):
        inputs = plt.imread(self.files[index % len(self.files)])
        vessels = plt.imread(self.vessels[index % len(self.vessels)])
        labels = plt.imread(self.files_using[index % len(self.files_using)])
        if self.transform is not None:
            inputs = self.transform(Image.fromarray(inputs))
            vessels = self.transform(Image.fromarray(vessels))
            labels = self.transform(Image.fromarray(labels))
        print(type(inputs))
        return inputs, vessels, labels

    def __len__(self):
        return len(self.files)
