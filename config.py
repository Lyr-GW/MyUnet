#IDRiD数据集根目录
# ROOT_DIR = 'D:/迅雷下载/idrid/A. Segmentation'
# ROOT_DIR = '/home/unet/data/A. Segmentation'
ROOT_DIR = '../data/A. Segmentation'

#训练集原始图片
ORIGIN_TRAIN_IMG = '/1. Original Images/a. Training Set'
TRAINING_PATH = '/home/unet/data/A. Segmentation/1. Original Images/a. Training Set'
#训练集视GroudTruth
ORIGIN_TRAIN_OD_GT = '/2. All Segmentation Groundtruths/a. Training Set/4. Soft Exudates'
TRAINING_GT_PATH = '/home/unet/data/A. Segmentation/1. Original Images/a. Training Set/2. All Segmentation ' \
                   'Groundtruths/a. Training Set/4. Soft Exudates '


# DDR_ROOT_DIR = '/home/linwei/UNet/data/lesion_segmentation'
DDR_ROOT_DIR = '../data/lesion_segmentation'

REF_IMG = '/ref_img.gif'
TEST_IMAGE_PATH = '../data/lesion_segmentation/valid/image/007-2846-100.jpg'

DDR_TRAIN_IMG = '/train/image'
DDR_TRAIN_VSL = '/train/vessels'
DDR_TRAIN_GT = '/train/label/HE'

DDR_VALID_IMG = '/valid/image' 
DDR_VALID_VSL = '/valid/vessels'
DDR_VALID_GT = '/valid/segmentation label/HE'

DDR_TEST_IMG = '/test/image' 
DDR_TEST_VSL = '/test/vessels'
DDR_TEST_GT = '/test/label/HE'

DDR_TEST_OUT = '/outputs/SE/test'


'''Hyper Parameters'''
BATCH_SIZE = 2
NUM_WORKERS = 2
LR = 0.0001
NUM_EPOCHS = 40

