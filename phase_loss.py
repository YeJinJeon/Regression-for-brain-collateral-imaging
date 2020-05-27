import torch
import torch.nn as nn

''' by long
class AveragePhaseLoss(nn.Module):

    def __init__(self):
        super(AveragePhaseLoss, self).__init__()
        self.loss = 0

    # input size: N, C, D, W, H
    def forward(self, predict, target, weight_mask, mask):
        predict_with_mask = torch.mul(predict, mask.repeat(1, 5, 1, 1, 1))
        target_with_mask = torch.mul(target, mask.repeat(1, 5, 1, 1, 1))
        absolute_err = target_with_mask - predict_with_mask
        matrix_loss = torch.mul(torch.mul(absolute_err, absolute_err), weight_mask).sum(dim=(2, 3, 4))
        elements_counts = mask.sum(dim=(2, 3, 4)) #total pixel number, 값이 1이므로
	    #total_number_of_pixel = mask[mask > 0].shape[0] = mask.sum(dim=(2,3,4))
        loss_of_all_phase = matrix_loss / elements_counts
        self.loss = loss_of_all_phase.mean()
        return self.loss
'''

def average_phase_loss(predict, target, weight_mask, mask):
    predict_with_mask = torch.mul(predict, mask.repeat(1, 5, 1, 1, 1))
    target_with_mask = torch.mul(target, mask.repeat(1, 5, 1, 1, 1))
    absolute_err = target_with_mask - predict_with_mask
    matrix_loss = torch.mul(torch.mul(absolute_err, absolute_err), weight_mask).sum(dim=(2, 3, 4))
    elements_counts = mask.sum(dim=(2, 3, 4))  # total pixel number, 값이 1이므로
    # total_number_of_pixel = mask[mask > 0].shape[0] = mask.sum(dim=(2,3,4))
    loss_of_all_phase = matrix_loss / elements_counts
    loss = loss_of_all_phase.mean()
    return loss