from torchvision import transforms
import albumentations
import albumentations.pytorch
from torch.utils.data import DataLoader
from dataset import IDRiDataset, DDRnonDataset, DDRDataset, DDRvDataset
from PIL import Image
import cv2
import config

RESIZE_SIZE = 768 or 512

### 数据增强参数
transform_args = albumentations.Compose([
    albumentations.Resize(512, 512), 
    albumentations.RandomCrop(448, 448),
    # albumentations.RandomCrop(224, 224),
    albumentations.CLAHE(),  
    albumentations.OneOf([
        albumentations.HorizontalFlip(p=0.7),
        albumentations.RandomRotate90(p=0.7),
        albumentations.VerticalFlip(p=0.7)            
    ], p=1),
    albumentations.pytorch.ToTensorV2()
])

test_transform_args = albumentations.Compose([
    albumentations.Resize(512, 512), 
    albumentations.CLAHE(),  
    albumentations.pytorch.ToTensorV2()
])


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
    # batch_size=1,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
)
#test 测试集
DDR_test_loader=DataLoader(
    dataset=DDRDataset(
        inputs_root=config.DDR_ROOT_DIR+config.DDR_TEST_IMG,
        # vessels_root=config.DDR_ROOT_DIR+config.DDR_TEST_VSL,  #血管信息
        labels_root=config.DDR_ROOT_DIR+config.DDR_TEST_GT,
        transform=test_transform_args,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
)