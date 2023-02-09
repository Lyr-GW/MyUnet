from torch import nn
import torch
import os

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    @staticmethod
    def forward(output, target, smooth=1e-5):
        y_pd = output.view(-1)
        y_gt = target.view(-1)
        intersection = torch.sum(y_pd * y_gt)
        score = (2. * intersection + smooth) / (torch.sum(y_pd) + torch.sum(y_gt) + smooth)
        loss = 1 - score
        return loss

class DiceBCELoss(nn.Module):
    def __init__(self, weight_ce=0.6):
        super(DiceBCELoss, self).__init__()
        self.weight_ce = weight_ce
        self.ce = nn.BCELoss()
        self.dc = DiceLoss()

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        result = self.weight_ce * ce_loss + (1 - self.weight_ce) * dc_loss
        return result

# def gram_matrix(y):
	# (ch, h, w) = y.size()
	# features = y.view(1, ch, w * h)
	# features_t = features.transpose(1, 2)
	# gram = features.bmm(features_t) / (ch * h * w)
	# return gram

def gram_matrix(y):
	(b, ch, h, w) = y.size()
	features = y.view(b, ch, w * h)
	features_t = features.transpose(1, 2)
	gram = features.bmm(features_t) / (ch * h * w)
	return gram

class StyleLoss(nn.Module):
	def __init__(self):
		super(StyleLoss, self).__init__()
		self.mse = nn.MSELoss()

	def forward(self, features, g_ref):
		loss = 0
		for feat, g_r in zip(features, g_ref):
			g = gram_matrix(feat)
			loss += self.mse(g, g_r)
		return loss