import time

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import torch.nn as nn
from loss import DiceLoss, DiceBCELoss, BCELoss, StyleLoss
import pytorch_ssim
import PIL.Image as Image
import torchvision
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# from medpy import metric
from torch.cuda import amp

import config
from dataloader import DDR_train_loader, DDR_valid_loader
# from unet import UNet
# from dual_unet import Dual_UNet
from triple_branches import Triple_Branches
import utils
import logging
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

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
# print(f"current using device --- {device}")
writer = SummaryWriter(log_dir="runs/0302")
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
        # print("Epoch:{}/{}".format(epoch+1, epochs))
        # print(train_loader.__len__())

        #设置训练模式
        model.train()
        train_loss = 0
        train_iou = 0
        train_step = 0
        val_loss = 0
        dice = 0
        val_dice = 0
        iou = 0
        val_iou = 0
        # precision = 0
        # accuracy = 0
        # val_precision = 0
        # val_accuracy = 0
        val_step = 0

        # for name, param in model.named_parameters():
        #     if param.grad is not None and torch.isnan(param.grad).any():
        #         print("nan gradient found")
        #         print("name:", name)

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            inputs = inputs.float()
            labels = labels.to(device)
            labels = labels.float()
            labels = utils.normalize(labels)

            # with amp.autocast(enabled=enable_amp):
            vsl_out, les_out = model(inputs)           #single input
            # les_out = model(inputs)           #single input
            # print("vsl_out shape"+str(vsl_out.shape))
            # print("les_out shape"+str(les_out.shape))
            # loss_style = style_loss(vsl_out.unsqueeze(0), gram_matrix(ref_img))
            # print(f"les_out.shape={les_out.shape},labels.shape={labels.shape}")
            # print(f"les_out={les_out}")
            # print(f"labels={labels}")
            loss_lesion = lesion_loss(les_out, labels)
            iou = utils.iou_score(les_out, labels)
            train_iou += iou
            # train_loss += loss_style.item() + loss_lesion.item()
            # print(f"loss_lesion shape--{loss_lesion.type}")
            train_loss += loss_lesion.item()
            train_step += 1

            '''自动混合精度'''
            # scaler.scale(train_loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            # loss = loss_style + loss_lesion
            # loss.backward()
            # with torch.autograd.detect_anomaly(): # 自动梯度检查
            loss_lesion.backward()
            # if (i+1) % 2 ==0 or (i+1) == len(train_loader):
                # if hasattr(torch.cuda, 'empty_cache'): torch.cuda.empty_cache()

            # logger.info(f"\n[Epoch {epoch+1}/{epochs}] [Batch {i+1}/{len(train_loader)}] [Lesion Loss: {loss_lesion.item()}]")
        
        #设置为验证模式
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(valid_loader):
                inputs = inputs.to(device)
                inputs = inputs.float()
                labels = labels.to(device)
                labels = labels.float()
                vsl_out, les_out = model(inputs)           #single input

                # loss_style = style_loss(vsl_out.unsqueeze(0), gram_matrix(ref_img))
                loss_lesion = lesion_loss(les_out, labels)
                # 计算IOU/Dice评价指标
                # dice = metric.dc(les_out.numpy(), labels.numpy())
                iou = utils.iou_score(les_out, labels)

                # # 计算准确度 精确度
                # accuracy = utils.accuracy(les_out, labels)
                # precision = utils.precision(les_out, labels)

                # loss = loss_style + loss_lesion
                # val_loss += loss_style.item() + loss_lesion.item()
                val_loss += loss_lesion.item()
                val_dice += dice
                val_iou += iou
                # val_precision += precision
                # val_accuracy += accuracy
                #统计训练loss
                val_step += 1
        # 分别求出整个epoch的训练loss以及验证指标
        train_loss /= train_step
        train_iou /= train_step
        val_loss /= val_step
        val_dice /= val_step
        val_iou /= val_step
        # val_precision /= val_step
        # val_accuracy /= val_step
        # print(f"\n[Epoch {epoch+1}/{epochs}] [Train Loss: {train_loss}] [Valid Loss: {val_loss}] [Lesion Accuracy: {val_accuracy}] [Lesion Precision: {val_precision}] ")
        # print(f"\n[Epoch {epoch+1}/{epochs}] [Train Loss: {train_loss}] [Valid Loss: {val_loss}] [Valid Lesion Dice: {val_dice}] [Valid Lesion IOU: {val_iou}]")
        # print(f"\n[Epoch {epoch+1}/{epochs}] [Train Loss: {train_loss}] [Valid Loss: {val_loss}] [Valid Lesion IOU: {val_iou}]")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        # scheduler.step(val_loss)
        logger.info(f"\n[Epoch {epoch+1}/{epochs}] [Train Loss: {train_loss}] [Train IoU: {train_iou}] [Valid Loss: {val_loss}] [Valid Lesion IoU: {val_iou}]")

        #tensorboard添加监视值
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', val_loss, epoch)
        writer.add_scalar('IoU/train', train_iou, epoch)
        writer.add_scalar('IoU/valid', val_iou, epoch)
        writer.close()
        # logger.info(f"\n[Epoch {epoch+1}/{epochs}] [Train Loss: {train_loss}] ")
        # 如果验证指标比最优值更好，那么保存当前模型参数
        if val_loss < best_score:
            best_score = val_loss
            torch.save(model.state_dict(), "./checkpoint.pth")
        
        epoch_end = time.time()
    logger.info('end logging.')


if __name__ == '__main__':
    # net = UNet(in_channels = 3, n_classes = 2)
    # net = Dual_UNet(in_ch = 3, vsl_ch = 4, out_ch = 2)
    net = Triple_Branches()
    net.to(device)
    # loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()
    # bce_loss = BCELoss()
    # iou_loss = 
    ssim_loss = pytorch_ssim.SSIM()
    style_loss = StyleLoss()
    dice_loss = DiceLoss()
    # loss = DiceBCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LR)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, eps=1e-9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,80], gamma=0.1)

    # ref_cpu = ref_img_tensor.cpu()
    # plt.imshow(ref_cpu)
    # plt.savefig('ref.jpg')

    train(net, DDR_train_loader, DDR_valid_loader, style_loss, bce_loss, ref_img_tensor, optimizer, scheduler, config.NUM_EPOCHS)