import torch
import numpy

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