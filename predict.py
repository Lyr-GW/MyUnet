import os
import torch
from torch.autograd import Variable
from PIL import Image
import numpy as np
from triple_branches import Triple_Branches
# from models.seg_net import Segnet
from dataloader import transform_args 
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = Segnet(3,2)
# model_path = './checkpoint/Segnet/model/netG_final.pth'

model = Triple_Branches()
model_path = './checkpoint.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
test_image_path = '../lesion_segmentation/valid/image/007-2846-100.jpg'
test_image = Image.open(test_image_path).convert('RGB')
print('Operating...')
img = transform_args(test_image)
img = img.unsqueeze(0)
img = Variable(img)
pred_vsl, pred_lesion  = model(img)
# pred_vsl.show()
predictions = pred_vsl.data.max(1)[1].squeeze_(1).cpu().numpy()
prediction = predictions[0]
# predictions_color = colorize_mask(prediction)
plt.imshow(prediction)
plt.savefig('vsl.jpg')
# prediction.show()
predictions = pred_lesion.data.max(1)[1].squeeze_(1).cpu().numpy()
prediction = predictions[0]
# predictions_color = colorize_mask(prediction)
plt.imshow(prediction)
plt.savefig('les.jpg')