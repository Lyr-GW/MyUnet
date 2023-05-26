import time

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import numpy as np
from loss import DiceLoss, DiceBCELoss, BCELoss, StyleLoss
import pytorch_ssim
import PIL.Image as Image
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# from medpy import metric
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

import config
from dataloader import DDR_train_loader, DDR_valid_loader
from unet import UNet
# from dual_unet import Dual_UNet
from triple_branches import Triple_Branches
import utils
import logging
from predict import predict_test
from torchviz import make_dot

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

# 部分全局对象
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda")
# print(f"current using device --- {device}")
now_time = time.strftime("%m%d.%H.%M", time.localtime())
writer = SummaryWriter(log_dir=f"new_runs/0526.15.30_{config.NUM_EPOCHS}epochs")
# 判断能否使用自动混合精度
# enable_amp = True if "cuda" in device.type else False
# 在训练最开始之前实例化一个GradScaler对象
# scaler = amp.GradScaler(enabled=enable_amp)

# 参考图片
ref_img = Image.open(config.DDR_ROOT_DIR+config.REF_IMG) 
ref_img_tensor = transforms.ToTensor()(ref_img)
ref_img_tensor = ref_img_tensor.unsqueeze(0)
# print(ref_img_tensor.shape)
ref_img_tensor = ref_img_tensor.to(device)


def gram_matrix(y):
	(b, ch, h, w) = y.size()
	features = y.view(b, ch, w * h)
	features_t = features.transpose(1, 2)
	gram = features.bmm(features_t) / (ch * h * w)
	return gram


def train(model, train_loader, valid_loader, style_loss, lesion_loss, ref_img, optimizer, scheduler, epochs):
    best_score = 1.0
    logger = get_logger('./log/exp.log')
    logger.info('start logging...')
    optimizer.zero_grad()

    for epoch in range(epochs):
        epoch_start = time.time()
        # print(f"------------epoch{epoch+1}-----------------")
        # print("Epoch:{}/{}".format(epoch+1, epochs))
        # print(train_loader.__len__())

        #设置训练模式
        model.train()
        train_loss = 0
        train_loss_style = 0
        train_loss_lesion = 0
        train_iou = 0
        train_step = 0
        val_loss = 0
        val_loss_style = 0
        val_loss_lesion = 0
        # dice = 0
        # val_dice = 0
        iou = 0
        val_iou = 0
        # precision = 0
        # accuracy = 0
        # val_precision = 0
        # val_accuracy = 0
        val_step = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            inputs = inputs.float()
            labels = labels.to(device)

            # with amp.autocast(enabled=enable_amp):

            vsl_out, les_out = model(inputs)           
            # # 模型结构可视化pdf
            # g = make_dot(model(inputs), params=dict(model.named_parameters()))        
            # g.view()

            # print(les_out.shape)
            labels = utils.binarize(labels)
            loss_style = style_loss(vsl_out.unsqueeze(0), gram_matrix(ref_img))
            loss_lesion = lesion_loss(les_out, labels)
            iou = utils.iou_score(les_out, labels)
            train_iou += iou
            # train_loss += loss_style.item() + loss_lesion.item()
            # train_loss += loss_lesion.item()

            '''自动混合精度'''
            # scaler.scale(train_loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            loss = loss_style + loss_lesion
            train_loss_style += loss_style
            train_loss_lesion += loss_lesion

            loss.backward()
            train_loss += loss.item()
            # loss_lesion.backward()

            # logger.info(f"\n[Epoch {epoch+1}/{epochs}] [Batch {i+1}/{len(train_loader)}] [Lesion Loss: {loss_lesion.item()}]")
        
        #设置为验证模式
        y_true = []
        y_pred = []
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(valid_loader):
                inputs = inputs.to(device)
                inputs = inputs.float()
                labels = labels.to(device)
                # labels = labels.float()
                vsl_out, les_out = model(inputs)           #single input
                labels = utils.binarize(labels)
                y_true.append(labels)
                y_pred.append(les_out)

                loss_style = style_loss(vsl_out.unsqueeze(0), gram_matrix(ref_img))
                loss_lesion = lesion_loss(les_out, labels)
                # 计算IOU评价指标
                iou = utils.iou_score(les_out, labels)

                # # 计算准确度 精确度
                # accuracy = utils.accuracy(les_out, labels)
                # precision = utils.precision(les_out, labels)

                # loss = loss_style + loss_lesion
                val_loss += loss_style.item() + loss_lesion.item()
                val_loss_lesion += loss_lesion.item()
                val_loss_style += loss_style.item()
                # val_loss += loss_lesion.item()
                # val_dice += dice
                val_iou += iou
                # val_precision += precision
                # val_accuracy += accuracy
                #统计训练loss
        # 分别求出整个epoch的训练loss以及验证指标
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        train_loss_style /= len(train_loader)
        train_loss_lesion /= len(train_loader)


        val_loss /= len(valid_loader)
        val_loss_style /= len(valid_loader)
        val_loss_lesion /= len(valid_loader)
        # val_dice /= len(valid_loader)
        val_iou /= len(valid_loader)
        if(epoch % 50 == 0):
            print(f'{epoch+1}epoch predicted')
            predict_test(f'epoch_{epoch+1}', config.DDR_ROOT_DIR + config.DDR_TEST_IMG + '/007-4679-200.jpg', model)
        # val_precision /= len(valid_loader)
        # val_accuracy /= len(valid_loader)
        # print(f"\n[Epoch {epoch+1}/{epochs}] [Train Loss: {train_loss}] [Valid Loss: {val_loss}] [Lesion Accuracy: {val_accuracy}] [Lesion Precision: {val_precision}] ")
        # print(f"\n[Epoch {epoch+1}/{epochs}] [Train Loss: {train_loss}] [Valid Loss: {val_loss}] [Valid Lesion Dice: {val_dice}] [Valid Lesion IOU: {val_iou}]")
        # print(f"\n[Epoch {epoch+1}/{epochs}] [Train Loss: {train_loss}] [Valid Loss: {val_loss}] [Valid Lesion IOU: {val_iou}]")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        # scheduler.step(val_loss)
        logger.info(f"\n\n[Epoch {epoch+1}/{epochs}]" + 
                    f" \n[Train Loss: {train_loss}] [Train_Style_Loss: {train_loss_style}] [Train_Lesion_Loss: {train_loss_lesion}] [Train IoU: {train_iou}]" + 
                    f" \n[Valid Loss: {val_loss}] [Valid_Style_Loss: {val_loss_style}] [Valid_Lesion_Loss: {val_loss_lesion}] [Valid Lesion IoU: {val_iou}]")

        # print("drawing roc curve")
        # if((epoch+1) >= 50 and (epoch+1) % 25 ==0):
        if((epoch+1) % 25 == 0):
            logger.info(f"epoch {epoch+1} drawing roc curve")
            fpr, tpr, roc_auc = utils.calculate_roc(model, valid_loader)
            # precisoin, recall = utils.calculate_pr(model, valid_loader)
            # # 使用tensorboard绘制roc曲线
            # writer.add_pr_curve('pr_curve', precisoin, recall, global_step = epoch)
            # writer.add_scalar('Curve/AP(average precision)', roc_auc, epoch)

            # 画ROC曲线
            plt.figure()
            lw = 2  # 线宽
            plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)  # 绘制ROC曲线
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  # 绘制对角线
            plt.xlim([0.0, 1.0])  # x轴的范围
            plt.ylim([0.0, 1.05])  # y轴的范围
            plt.xlabel('False Positive Rate')  # x轴的标签
            plt.ylabel('True Positive Rate')  # y轴的标签
            plt.title('Receiver operating characteristic example')  # 标题
            plt.legend(loc="lower right")  # 图例
            # plt.show()  # 展
            # plt.draw()
            # plt_obj = plt.gcf()  # 获取当前figure
            # # image = np.frombuffer(plt_obj.canvas.tostring_rgb(), dtype='uint8')  # 转换为numpy数组
            # # image = image.reshape(plt_obj.canvas.get_width_height()[::-1] + (3,))  # 调整形状
            # # 创建一个画布对象
            # canvas = plt_obj.canvas
            # # 将图像绘制到画布
            # canvas.draw()
            # # 获取画布上的像素数据
            # width, height = plt_obj.canvas.get_width_height()
            # image = np.frombuffer(plt_obj.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)

            # writer.add_image('roc_curve', image)
            # writer.add_scalar('Curve/ROC_AUC', roc_auc, epoch)
            if(os.path.exists(f"./result/roc/{now_time}") == False):
                os.makedirs(f"./result/roc/{now_time}")
            plt.savefig(f"./result/roc/{now_time}/epoch_{epoch+1}.jpeg")
            # writer.add_image('roc_curve', "./result/roc/epoch_5.jpeg")
            writer.add_scalar('Curve/ROC_AUC', roc_auc, epoch)
            print("roc done")

        #tensorboard添加监视值
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/train_lesion', train_loss_lesion, epoch)
        writer.add_scalar('Loss/train_style', train_loss_style, epoch)
        writer.add_scalar('Loss/valid', val_loss, epoch)
        writer.add_scalar('Loss/valid_lesion', val_loss_lesion, epoch)
        writer.add_scalar('Loss/valid_style', val_loss_style, epoch)
        writer.add_scalar('IoU/train', train_iou, epoch)
        writer.add_scalar('IoU/valid', val_iou, epoch)
        # writer.flush()
        # logger.info(f"\n[Epoch {epoch+1}/{epochs}] [Train Loss: {train_loss}] ")
        # 如果验证指标比最优值更好，那么保存当前模型参数
        if val_loss < best_score:
            logger.info(f"best epoch--{epoch+1}, val_loss--{val_loss}, val_iou--{val_iou}")
            best_score = val_loss
            torch.save(model.state_dict(), "./checkpoint.pth")
        
        epoch_end = time.time()
    logger.info('end logging.')


if __name__ == '__main__':
    # net = UNet(n_channels = 3, n_classes = 1)
    # net = Dual_UNet(in_ch = 3, vsl_ch = 4, out_ch = 2)
    net = Triple_Branches()
    net.to(device)
    # print(net)
    # loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()
    # bce_loss = BCELoss()
    # iou_loss = 
    ssim_loss = pytorch_ssim.SSIM()
    style_loss = StyleLoss()
    dice_loss = DiceLoss()
    # loss = DiceBCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LR)
    # optimizer = torch.optim.SGD(net.parameters(), lr=config.LR, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, eps=1e-9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,140], gamma=0.1)

    # ref_cpu = ref_img_tensor.cpu()
    # plt.imshow(ref_cpu)
    # plt.savefig('ref.jpg')

    train(net, DDR_train_loader, DDR_valid_loader, style_loss, bce_loss, ref_img_tensor, optimizer, scheduler, config.NUM_EPOCHS)
    writer.close()