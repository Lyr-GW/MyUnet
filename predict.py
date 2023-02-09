import os
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

import config

transform_args = transforms.Compose([
    transforms.Resize([512, 512]),
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(index, img):
    model = Triple_Branches()
    model_path = 'D:/Study/Code/unet/code/checkpoint.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    # test_image_path = config.TEST_IMAGE_PATH
    print('Operating...')

    plt.figure("cell")
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    # print (img.shape)  
    # print (img.dtype )
    # print (img.size) 
    # print (type(img))
    # print (img.shape[2])
    img = transform_args(img)
    img = img.unsqueeze(0)
    # img = img.to(device)
    img = Variable(img)

    model.eval()
    with torch.no_grad():
        pred_vsl, pred_lesion  = model(img)
        # pred_vsl.show()
        predictions = pred_vsl.data.max(1)[1].squeeze_(1).cpu().numpy()
        prediction = predictions[0]
        # predictions_color = colorize_mask(prediction)
        plt.imshow(prediction)
        plt.savefig(f'./result/plt/{index}_vsl.jpg')
        # prediction.show()
        predictions = pred_lesion.data.max(1)[1].squeeze_(1).cpu().numpy()
        prediction = predictions[0]
        # predictions_color = colorize_mask(prediction)
        plt.imshow(prediction)
        plt.savefig(f'./result/plt/{index}_les.jpg')

        pred_vsl = torch.squeeze(pred_vsl)                      # 将(batch、channel)维度去掉
        pred_vsl = np.array(pred_vsl.data.cpu())                # 保存图片需要转为cpu处理
    
        pred_vsl[pred_vsl >=0 ] =0                            # 转为二值图片
        pred_vsl[pred_vsl < 0 ] =255
 
        pred_vsl = np.uint8(pred_vsl)                           # 转为图片的形式
        cv2.imwrite(f'./result/cv2/{index}_vsl.png', pred_vsl)           # 保存图片

        pred_lesion = torch.squeeze(pred_lesion)                      # 将(batch、channel)维度去掉
        pred_lesion = np.array(pred_lesion.data.cpu())                # 保存图片需要转为cpu处理
 
        pred_lesion[pred_lesion >=0 ] =0                            # 转为二值图片
        pred_lesion[pred_lesion < 0 ] =255
 
        pred_lesion = np.uint8(pred_lesion)                           # 转为图片的形式
        cv2.imwrite(f'./result/cv2/{index}_les.png', pred_lesion)           # 保存图片

        # # img = torch.squeeze(img)                      # 将(batch、channel)维度去掉
        # img = np.array(img.data.cpu())                # 保存图片需要转为cpu处理
        # img = np.uint8(img)
        # print(img.shape)
        # cv2.imwrite('D:/Study/Code/unet/code/result/origin.png', img)           # 保存图片
    print('Done')



if __name__ == '__main__':
    test_imgs = os.listdir(config.DDR_ROOT_DIR + config.DDR_TEST_IMG)
    for item in test_imgs:
        img_path = config.DDR_ROOT_DIR + config.DDR_TEST_IMG + '/' + item
        print(img_path)
        img = Image.open(img_path)
        predict(item, img)
        # print(type(item))
        # predict(item.name, )