import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
from triple_branches import Triple_Branches
# from models.seg_net import Segnet
from dataloader import transform_args 
import matplotlib.pyplot as plt
import numpy as np
import cv2

from dataloader import DDR_test_loader
import config
import utils

transform_args = transforms.Compose([
    transforms.Resize([640, 640]),
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(index, img):
    model = Triple_Branches()
    model_path = './checkpoint.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    # test_image_path = config.TEST_IMAGE_PATH
    print('Operating...')

    img = transform_args(img)
    img = img.unsqueeze(0)
    # img = img.to(device)
    img = Variable(img)

    model.eval()


    with torch.no_grad():
        pred_vsl, pred_lesion  = model(img)
        # predictions = pred_vsl.data.max(1)[1].squeeze_(1).cpu().numpy()
        # prediction = predictions[0]
        # predictions_color = colorize_mask(prediction)
        # plt.imshow(prediction)
        # plt.savefig(f'./result/plt/{index}_vsl.jpg')
        # prediction.show()

        '''评价指标'''
        # iou = utils.iou_score(pred_lesion, labels)


        predictions = pred_lesion.data.max(1)[1].squeeze_(1).cpu().numpy()
        prediction = predictions[0]
        prediction *= 255
        # predictions_color = colorize_mask(prediction)
        plt.imshow(prediction)
        plt.savefig(f'./result/plt/{index}_les.jpg')

        pred_lesion = torch.squeeze(pred_lesion)                      # 将(batch、channel)维度去掉
        pred_lesion = np.array(pred_lesion.data.cpu())                # 保存图片需要转为cpu处理
 
        pred_lesion[pred_lesion >=0.4 ] =255                            # 转为二值图片
        pred_lesion[pred_lesion < 0.4 ] =0
 
        pred_lesion = np.uint8(pred_lesion)                           # 转为图片的形式
        cv2.imwrite(f'./result/cv2/{index}_les.png', pred_lesion)           # 保存图片

        # pred_vsl = torch.squeeze(pred_vsl)                      # 将(batch、channel)维度去掉
        # pred_vsl = np.array(pred_vsl.data.cpu())                # 保存图片需要转为cpu处理
    
        # pred_vsl[pred_vsl >=0 ] =0                            # 转为二值图片
        # pred_vsl[pred_vsl < 0 ] =255
 
        # pred_vsl = np.uint8(pred_vsl)                           # 转为图片的形式
        # cv2.imwrite(f'./result/cv2/{index}_vsl.png', pred_vsl)           # 保存图片

        # # img = torch.squeeze(img)                      # 将(batch、channel)维度去掉
        # img = np.array(img.data.cpu())                # 保存图片需要转为cpu处理
        # img = np.uint8(img)
        # print(img.shape)
        # cv2.imwrite('D:/Study/Code/unet/code/result/origin.png', img)           # 保存图片
    print('Done')

def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(1, 2, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
        
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)

def test_uni(test_loader):
    for i, (inputs, labels) in enumerate(test_loader):
        inputs.squeeze(0)
        inputs = inputs.cpu()
        inputs = inputs.float()
        labels = labels.squeeze(dim=0)
        labels = labels.squeeze(dim=0)

        labels.squeeze(0)
        labels = labels.cpu()
        labels = labels.float()
        labels = labels.squeeze(dim=0)
        labels = labels.squeeze(dim=0)
        _labels = utils.normalize(labels)

        print("draw origin image")
        labels = np.uint8(labels)
        cv2.imwrite(f'./result/label_test/cv2_{i}_les.png', labels)           # 保存图片

        plt.imshow(labels)
        plt.savefig(f'./result/label_test/plt_{i}_les.jpg')
            
        print("draw normalized image")
        _labels = np.uint8(_labels)
        cv2.imwrite(f'./result/label_test/cv2_{i}_les_n.png', _labels)           # 保存图片

        plt.imshow(_labels)
        plt.savefig(f'./result/label_test/plt_{i}_les_n.jpg')


        
iou = 0
test_iou = 0
test_step = 0

def test(test_loader):
    model = Triple_Branches()
    model.to(device)
    model_path = './checkpoint.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    iou = 0
    test_iou = 0
    test_step = 0
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            inputs = inputs.float()
            labels = labels.to(device)
            labels = labels.float()
            pred_vsl, pred_lesion = model(inputs)

            iou = utils.iou_score(pred_lesion, labels)
            test_iou += iou
            test_step += 1

            '''图片可视化'''
            # predictions = pred_lesion.data.max(1)[1].squeeze_(1).cpu().numpy()
            # prediction = predictions[0]
            # # prediction *= 255
            # # predictions_color = colorize_mask(prediction)
            # plt.imshow(prediction)
            # plt.savefig(f'./result/plt/{i}_les.jpg')

            # print(pred_lesion)
            pred_lesion = utils.normalize(pred_lesion)
            pred_lesion *= 255
            # print(pred_lesion)
            pred_lesion = torch.squeeze(pred_lesion)                      # 将(batch、channel)维度去掉
            pred_lesion = np.array(pred_lesion.data.cpu())                # 保存图片需要转为cpu处理
 
            # pred_lesion[pred_lesion >=0.6 ] =255                            # 转为二值图片
            # pred_lesion[pred_lesion < 0.6 ] =0
            # print(pred_lesion)
 
            pred_lesion = np.uint8(pred_lesion)                           # 转为图片的形式
            # print(pred_lesion.shape)
            cv2.imwrite(f'./result/cv2/{i}_les.png', pred_lesion)           # 保存图片

        
        test_iou /= test_step
        print(f"test_iou={test_iou}")





if __name__ == '__main__':
    # test_uni(DDR_test_loader)
    test(DDR_test_loader)

    # test_imgs = os.listdir(config.DDR_ROOT_DIR + config.DDR_TEST_IMG)
    # # item = '007-4679-200.jpg'
    # # img_path = config.DDR_ROOT_DIR + config.DDR_TEST_IMG + '/' + item
    # # print(img_path)
    # # img = Image.open(img_path)
    # # predict(item, img)
    # for item in test_imgs:
    #     img_path = config.DDR_ROOT_DIR + config.DDR_TEST_IMG + '/' + item
    #     print(img_path)
    #     img = Image.open(img_path)
    #     predict(item, img)
    #     # print(type(item))
    #     # predict(item.name, )