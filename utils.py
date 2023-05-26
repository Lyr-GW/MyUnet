import torch
import numpy as np 
import cv2
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.utils.multiclass import type_of_target
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 归一到[0,1]区间
def normalize(x):
    max_x = torch.max(x)
    min_x = torch.min(x)
    # print(x)
    n = (x - min_x) / (max_x - min_x + 1e-6)
    return n

# 二值化
def binarize(x):
    max_x = torch.max(x)
    min_x = torch.min(x)
    # print(x)
    n = (x - min_x) / (max_x - min_x + 1e-6)
    n[n >=0.5 ] = 1                            # 转为二值图片
    n[n < 0.5 ] = 0
    # print(f"max--{max_x}, min--{min_x}, max-min--{max_x-min_x+1e-6}, n--{n}")
    # n[n >=0.4 ] =255                            # 转为二值图片
    # n[n < 0.4 ] =0
    return n

def precision(predict, target):
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = np.atleast_1d(predict.astype(np.bool))
    target = np.atleast_1d(target.astype(np.bool))

    tp = np.count_nonzero(predict & target)
    tn = np.count_nonzero(~predict & ~target)
    fp = np.count_nonzero(predict & ~target)
    fn = np.count_nonzero(~predict & target)

    try:
        precision = tp / float(tp + fp)
    except ZeroDivisionError:
        precision = 0.0

    return precision


def accuracy(predict, target):
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = np.atleast_1d(predict.astype(np.bool))
    target = np.atleast_1d(target.astype(np.bool))

    tp = np.count_nonzero(predict & target)
    tn = np.count_nonzero(~predict & ~target)
    fp = np.count_nonzero(predict & ~target)
    fn = np.count_nonzero(~predict & target)


    try:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError:
        accuracy = 0.0

    # print(f"accuracy--{accuracy}\t|tp--{tp},tn--{tn},fp--{fp},fn--{fn}|float(tp + tn + fp + fn)--{float(tp + tn + fp + fn)}")
    return accuracy

def toImage(name, path, img):
    img = torch.squeeze(img)                      # 将(batch、channel)维度去掉
    img = np.array(img.data.cpu()) 
    # threshold = 0.4               # 保存图片需要转为cpu处理
 
    # img[img >= threshold ] =255                            # 转为二值图片
    # img[img < threshold ] =0
 
    img = np.uint8(img)                           # 转为图片的形式
    cv2.imwrite(f'{path}/{name}.png', img)           # 保存图片

#IOU
def iou_score(output, target):
    smooth = 1e-7

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    # # print(f"iou__output_={output_}")
    # toImage("iou_output_", "./result/iou", torch.tensor(output_*255))
    # # print(f"iou__target_={target_}")
    # toImage("iou_target_", "./result/iou", torch.tensor(target_*255))
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

def iou(predict, label):
    smooth = 1e-10
    intersection = np.multiply(predict, label)
    union = np.asarray(predict+label>0, np.float32)
    iou = intersection / (union + smooth)
    return iou

def calculate_roc(model, dataloader):
    model.eval()
    y_score = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            inputs = inputs.float()
            labels = labels.to(device)
            labels = labels.float()
            vsl_out, les_out = model(inputs)
            y_score.append(les_out.cpu().numpy())
            y_true.append(labels.cpu().numpy())

    y_score = np.concatenate(y_score, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    y_true = y_true.reshape(-1, 1)  #铺平
    y_score = y_score.reshape(-1, 1)
    # print('y_score shape:', y_score.shape)
    # print('y_true shape:', y_true.shape)
    y_score = y_score.tolist()  #转成list
    y_true = y_true.tolist()
    # print(type_of_target(y_score))
    # print(type_of_target(y_true))
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc

def calculate_pr(model, dataloader):
    model.eval()
    y_score = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            inputs = inputs.float()
            labels = labels.to(device)
            labels = labels.float()
            vsl_out, les_out = model(inputs)
            y_score.append(les_out.cpu().numpy())
            y_true.append(labels.cpu().numpy())

    y_score = np.concatenate(y_score, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    y_true = y_true.reshape(-1, 1)  #铺平
    y_score = y_score.reshape(-1, 1)
    print('y_score shape:', y_score.shape)
    print('y_true shape:', y_true.shape)
    y_score = y_score.tolist()  #转成list
    y_true = y_true.tolist()
    # print(type_of_target(y_score))
    # print(type_of_target(y_true))
    precision, recall, _ = precision_recall_curve(y_true, y_score)    

    return precision, recall

def roc_c(y_true, y_pred):
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
    # 计算FPR和TPR
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    # 计算AUC
    roc_auc = auc(fpr, tpr)

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
    plt.show()  # 展示
    #增加保存图像代码