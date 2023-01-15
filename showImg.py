import os

import config
from dataloader import train_loader

path = '/home/unet/data/A. Segmentation/1. Original Images/a. Training Set'

if __name__ == '__main__':
    for i, (inputs, labels) in enumerate(train_loader):
        print(len(labels))
        # print("inputs||" + str(inputs) + '\r')
    # imgs = os.listdir(path)
    imgs = os.listdir(config.ROOT_DIR+config.ORIGIN_TRAIN_IMG)
    print('len of files in dir: ' + str(len(imgs)))
