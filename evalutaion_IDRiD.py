
import os
import numpy as np
from PIL import Image

from sklearn.metrics import auc

def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path) if fname.endswith(extension)]
    
    if sort:
        filenames = ns.natsorted(filenames)
    
    return filenames

def pr_metric(true_img, pred_img):
    """
    Precision-recall curve
    """
    precision, recall, thresholds = precision_recall_curve(true_img.flatten(), pred_img.flatten())
    AUC_prec_rec = auc(recall, precision)
    
    # compute bset f1
    best_f1 = -1
    for index in range(len(precision)):
        curr_f1 = 2.*precision[index] * recall[index] / (precision[index] + recall[index])
        if best_f1 < curr_f1:
            best_f1 = curr_f1
            prec = precision[index]
            sen = recall[index]
            best_threshold = thresholds[index]
    return AUC_prec_rec, best_f1, best_threshold, sen, prec, precision, recall

def image_shape(filename):
    img = Image.open(filename)
    img_arr = np.asarray(img)
    img_shape = img_arr.shape
    return img_shape

def imagefiles2arrs(filenames, augment=False, z_score=False):
    img_shape = image_shape(filenames[0])
    if len(img_shape) == 3:
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
    elif len(img_shape) == 2:
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1]), dtype=np.float32)
    # convert to z score per image
    for file_index in range(len(filenames)):
        im = Image.open(filenames[file_index])
        img = np.array(im)
        images_arr[file_index] = img
    return images_arr

if __name__ == '__main__':
    predicated_mask_dir = 'predicated_mask_dir'
    ground_truth_dir = 'ground_truth_dir'
    predicated_mask_filenames = all_files_under(predicated_mask_dir)
    ground_truth_filenames = all_files_under(ground_truth_dir)

    predicated_all = imagefiles2arrs(predicated_mask_filenames) / 255.
    index_gt = 0
    ground_truth_all = np.zeros(predicated_all.shape)

    for index_predicated in range(len(predicated_mask_filenames)):
        # build array of gt
        if index_gt < len(ground_truth_filenames) and os.path.basename(predicated_mask_filenames[index_predicated]).replace(".jpg", "") in os.path.basename(ground_truth_filenames[index_gt]):
            ground_truth = imagefiles2arrs(ground_truth_filenames[index_gt:index_gt + 1]).astype(np.uint8)[0, ...]
            ground_truth_all[index_predicated, ...] = ground_truth
            index_gt += 1
    aupr_test, best_f1_test, best_f1_thresh_test, sen_test, ppv_test= pr_metric(ground_truth_all, predicated_all)
    print(aupr_test)

