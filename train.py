import time

import torch
import torch.nn as nn
from loss import DiceLoss, DiceBCELoss, StyleLoss
import pytorch_ssim
import PIL.Image as Image
import torchvision.transforms as transforms

import config
from dataloader import train_loader, DDR_train_loader, DDR_valid_loader
from unet import UNet
from dual_unet import Dual_UNet
from triple_branches import Triple_Branches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参考图片
ref_img = Image.open(config.DDR_ROOT_DIR+config.REF_IMG) 
ref_img_tensor = transforms.ToTensor()(ref_img)
ref_img_tensor = ref_img_tensor.unsqueeze(0)
print(ref_img_tensor.shape)
ref_img_tensor = ref_img_tensor.to(device)


def gram_matrix(y):
	(b, ch, h, w) = y.size()
	features = y.view(b, ch, w * h)
	features_t = features.transpose(1, 2)
	gram = features.bmm(features_t) / (ch * h * w)
	return gram

def train(model, train_loader, valid_loader, style_loss, lesion_loss, ref_img, optimizer, epochs):
    best_score = 1.0
    for epoch in range(epochs):
        epoch_start = time.time()
        # print("Epoch:{}/{}".format(epoch+1, epochs))
        # print(train_loader.__len__())

        #设置训练模式
        model.train()
        train_loss = 0
        train_step = 0
        val_loss = 0
        val_step = 0

        print(len(train_loader))
        print(train_loader.dataset)
        # for i, (inputs, vessels, labels) in enumerate(train_loader):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # labels = labels.long()
            # vessels = vessels.to(device)
            # print(vessels.shape)
            # print(vessels[0].shape)
            # inputs = torch.FloatTensor(inputs)
            optimizer.zero_grad()
            vsl_out, les_out = model(inputs)           #single input
            # out = model(inputs, vessels)    #dual inputs

            # print('预测值：', out.shape)  #shape--(batch_size,channels,height,width)
            # print('标签值： ', labels[:,:1].squeeze(1).shape)  
            # if model.n_classes > 1:
            # if model.out_ch > 1:
            #     loss = loss_function(out, labels[:,:1].squeeze(1))
            # else:
            #     loss = loss_function(out, labels[:,:1])
            # print(ref_img.shape)
            # print(vsl_out.shape)
            loss_style = style_loss(vsl_out, gram_matrix(ref_img))
            loss_lesion = lesion_loss(les_out, labels)
            # loss_lesion = lesion_loss(les_out, labels.repeat(1,2,1,1))
            #统计训练loss
            #12.27  如何综合血管和病灶loss
            train_loss += loss_style.item() + loss_lesion.item()
            train_step += 1

            # loss_style.backward()
            # loss_lesion.requires_grad_(True)
            # loss_lesion.backward(retain_graph=True)
            loss = loss_style + loss_lesion
            loss.backward()
            optimizer.step()
            print("\n[Epoch {}/{}] [Batch {}/{}] [Vessel loss: {}] [Lesion loss: {}]".format(epoch + 1, epochs, i + 1, len(train_loader), loss_style.item(), loss_lesion.item()))
            # with open(r"../data/losses/loss.txt", "a") as f1:
            #     f1.write("\n[Epoch {}/{}] [Batch {}/{}] [D loss: {}]".format(epoch + 1, epochs, i, len(train_loader), loss.item()))
            #     f1.close()
            # torch.save(model, "../saved_models/Unet_%d.pkl" % epoch)
        
        #设置为验证模式
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(valid_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # labels = labels.long()
                # vessels = vessels.to(device)
            
                # out = model(inputs, vessels)    #dual inputs
                vsl_out, les_out = model(inputs)           #single input
                #统计评价loss
                # val_loss += DiceLoss()(model(inputs, vessels), labels).item()
                # if model.out_ch > 1:
                #     loss = loss_function(out, labels[:,:1].squeeze(1))
                # else:
                #     loss = loss_function(out, labels[:,:1])

                loss_style = style_loss(vsl_out, gram_matrix(ref_img))
                loss_lesion = lesion_loss(les_out, labels)
                # loss_lesion = lesion_loss(les_out, labels.repeat(1,2,1,1))

                #统计训练loss
                #12.27  如何综合血管和病灶loss
                train_loss += loss_style.item() + loss_lesion.item()
                train_step += 1


                loss = loss_style + loss_lesion
                # loss.backward()

                # val_loss += loss
                val_loss += loss_style.item() + loss_lesion.item()
            #统计训练loss
                val_step += 1
        # 分别求出整个epoch的训练loss以及验证指标
        train_loss /= train_step
        val_loss /= val_step
        print(f"\n[Epoch {epoch+1}/{epochs}] [Train Loss: {train_loss}] [Valid Loss: {val_loss}]")
        # 如果验证指标比最优值更好，那么保存当前模型参数
        if val_loss < best_score:
            best_score = val_loss
            torch.save(model.state_dict(), "./checkpoint.pth")

        epoch_end = time.time()


if __name__ == '__main__':
    # net = UNet(in_channels = 3, n_classes = 2)
    # net = Dual_UNet(in_ch = 3, vsl_ch = 4, out_ch = 2)
    net = Triple_Branches()
    net.to(device)
    loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()
    # iou_loss = 
    ssim_loss = pytorch_ssim.SSIM()
    style_loss = StyleLoss()
    dice_loss = DiceLoss()
    # loss = DiceBCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LR)

    train(net, DDR_train_loader, DDR_valid_loader, style_loss, bce_loss, ref_img_tensor, optimizer, config.NUM_EPOCHS)