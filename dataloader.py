from torchvision import transforms
import albumentations
import albumentations.pytorch
from torch.utils.data import DataLoader
from dataset import IDRiDataset, DDRnonDataset, DDRDataset, DDRvDataset
from PIL import Image
import config


transform_args = albumentations.Compose([
    # transforms.Resize([512, 512]),
    # transforms.ToTensor(),
    albumentations.Resize(1024,1024), 
    # albumentations.RandomCrop(224, 224),
    albumentations.OneOf([
        albumentations.HorizontalFlip(p=1),
        albumentations.RandomRotate90(p=1),
        albumentations.VerticalFlip(p=1)            
    ], p=1),
    albumentations.OneOf([
        albumentations.MotionBlur(p=1),
        albumentations.OpticalDistortion(p=1),
        albumentations.GaussNoise(p=1)                 
    ], p=1),
    albumentations.pytorch.transforms.ToTensorV2()
])

#训练集loader
train_loader=DataLoader(
    dataset=IDRiDataset(
        inputs_root=config.ROOT_DIR+config.ORIGIN_TRAIN_IMG,
        labels_root=config.ROOT_DIR+config.ORIGIN_TRAIN_OD_GT,
        # inputs_root=config.TRAINING_PATH,
        # labels_root=config.TRAINING_GT_PATH,
        transform=transform_args,
    ),
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
)

#DDR loader
#train 训练集
DDR_train_loader=DataLoader(
    # dataset=DDRvDataset(
    dataset=DDRDataset(
        inputs_root=config.DDR_ROOT_DIR+config.DDR_TRAIN_IMG,
        labels_root=config.DDR_ROOT_DIR+config.DDR_TRAIN_GT,
        transform=transform_args,
    ),
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
)
#valid 验证集
DDR_valid_loader=DataLoader(
    dataset=DDRDataset(
        inputs_root=config.DDR_ROOT_DIR+config.DDR_VALID_IMG,
        # vessels_root=config.DDR_ROOT_DIR+config.DDR_VALID_VSL,  #血管信息
        labels_root=config.DDR_ROOT_DIR+config.DDR_VALID_GT,
        transform=transform_args,
    ),
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
)
#test 测试集
DDR_test_loader=DataLoader(
    dataset=DDRDataset(
        inputs_root=config.DDR_ROOT_DIR+config.DDR_TEST_IMG,
        # vessels_root=config.DDR_ROOT_DIR+config.DDR_TEST_VSL,  #血管信息
        labels_root=config.DDR_ROOT_DIR+config.DDR_TEST_GT,
        transform=transform_args,
    ),
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
)