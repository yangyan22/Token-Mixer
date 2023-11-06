import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):  # only calculate the loss in target words within masks
        # truncate to the same size
        # print(input.shape) # BS, Max Length of a batch, 761==vocab_size + 1
        target = target[:, :input.size(1)]  # BS, Length 314, 624, 752,  ...,   0,   0,   0
        mask = mask[:, :input.size(1)]  # BS, Length  1., 1., 1.,  ..., 0., 0., 0.
        output = -input.gather(dim=2, index=target.long().unsqueeze(2)).squeeze(2) * mask
        # gather: get index=target values in input (softmax-log outputs)
        output = torch.sum(output) / torch.sum(mask)
        return output


def compute_loss(prediction, reports_ids, reports_masks):
    criterion = LanguageModelCriterion()
    loss = criterion(prediction, reports_ids, reports_masks)
    return loss
