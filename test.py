import torch
from PIL import Image
import imageio
import numpy as np
import os

import config
from dual_unet import Dual_UNet 
# 设置使用的显卡编号
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def test(img_dir, vsl_dir, weight_path, outputs_dir, img_size=(512, 512)):
    """
    :param img_dir:需要预测的数据的原始图片文件夹
    :param vsl_dir:需要预测的数据的血管图片文件夹
    :param weight_path: 权重文件路径
    :param outputs_dir: 输出文件夹
    :param img_size: 图片大小
    :return:
    """
    # 定义网络结构，并且加载到显卡
    network = Dual_UNet(in_ch = 3, vsl_ch = 4, out_ch = 2)
    network = network.cuda()
    # 加载权重文件（训练好的网络）
    network.load_state_dict(torch.load(weight_path))
    # 获取测试文件夹的文件
    file_list = os.listdir(img_dir)
    for f in file_list:
        # 读取图片并完成缩放
        # print(f[:f.index('.')])
        img = np.array(Image.open(os.path.join(img_dir, f)).resize(img_size, Image.BILINEAR))
        tif = f[:f.index('.')] + '.tif'
        vsl = np.array(Image.open(os.path.join(vsl_dir, tif)).resize(img_size, Image.BILINEAR))
        # vsl = imageio.imread(os.path.join(vsl_dir, tif))
        # vsl = np.array(Image.open(os.path.join(vsl_dir, tif)))
        # 增加batch维度
        print(vsl.shape)
        img = np.expand_dims(img, axis=0)
        vsl = np.expand_dims(vsl, axis=0)
        vsl = np.expand_dims(vsl, axis=3)
        # print(img)
        # print(vsl)
        # 更改通道顺序（BHWC->BCHW）
        img = img.transpose((0, 3, 1, 2))
        vsl = vsl.transpose((0, 3, 1, 2))
        # 转为浮点类型
        img = img.astype(np.float32)
        vsl = vsl.astype(np.float32)
        img_cuda = torch.from_numpy(img).cuda()
        vsl_cuda = torch.from_numpy(vsl).cuda()
        # 预测结果并且从显存转移到内存中
        pred = network(img_cuda, vsl_cuda)
        pred = pred.clone().cpu().detach().numpy()
        # 二值化操作
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        # 保存结果到输出文件夹
        print(type(Image.fromarray(pred[0, 0, :, :])))
        Image.fromarray(pred[0, 0, :, :]).save(os.path.join(outputs_dir, f))


if __name__ == "__main__":
    # test("./data/test/images", "./checkpoint.pth", "./data/test/outputs")
    img_dir = config.DDR_ROOT_DIR +config.DDR_TEST_IMG
    vsl_dir = config.DDR_ROOT_DIR +config.DDR_TEST_VSL
    outputs_dir = config.DDR_ROOT_DIR + config.DDR_TEST_OUT
    test(img_dir, vsl_dir, "./checkpoint.pth", outputs_dir)
