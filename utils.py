import torch
import numpy 

def normalize(x):
    max_x = torch.max(x)
    min_x = torch.min(x)
    # print(x)
    n = (x - min_x) / (max_x - min_x + 1e-6)
    n[n >=0.4 ] = 1                            # 转为二值图片
    n[n < 0.4 ] = 0
    # print(f"max--{max_x}, min--{min_x}, max-min--{max_x-min_x+1e-6}, n--{n}")
    # n[n >=0.4 ] =255                            # 转为二值图片
    # n[n < 0.4 ] =0
    return n

def precision(predict, target):
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = numpy.atleast_1d(predict.astype(numpy.bool))
    target = numpy.atleast_1d(target.astype(numpy.bool))

    tp = numpy.count_nonzero(predict & target)
    tn = numpy.count_nonzero(~predict & ~target)
    fp = numpy.count_nonzero(predict & ~target)
    fn = numpy.count_nonzero(~predict & target)

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

    predict = numpy.atleast_1d(predict.astype(numpy.bool))
    target = numpy.atleast_1d(target.astype(numpy.bool))

    tp = numpy.count_nonzero(predict & target)
    tn = numpy.count_nonzero(~predict & ~target)
    fp = numpy.count_nonzero(predict & ~target)
    fn = numpy.count_nonzero(~predict & target)


    try:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError:
        accuracy = 0.0

    # print(f"accuracy--{accuracy}\t|tp--{tp},tn--{tn},fp--{fp},fn--{fn}|float(tp + tn + fp + fn)--{float(tp + tn + fp + fn)}")
    return accuracy

#IOU
def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

def iou(predict, label):
    smooth = 1e-10
    intersection = numpy.multiply(predict, label)
    union = numpy.asarray(predict+label>0, numpy.float32)
    iou = intersection / (union + smooth)
    return iou