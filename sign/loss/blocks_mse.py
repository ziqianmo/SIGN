import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn
from torch.autograd import Variable



class blocks_mse(nn.Module):

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.device = device


    def forward(self, image_features1, image_features2):

        # 1. concate
        if 1:

            channels1 = [image_features1[:, :, i] for i in range(image_features1.shape[2])]
            channels2 = [image_features2[:, :, i] for i in range(image_features2.shape[2])]

            # 使用 torch.cat 连接所有通道
            image_features_blocks_1 = torch.cat(channels1, dim=-1)
            image_features_blocks_2 = torch.cat(channels2, dim=-1)

            image_features1 = F.normalize(image_features_blocks_1, dim=-1)
            image_features2 = F.normalize(image_features_blocks_2, dim=-1)

            loss = torch.nn.functional.mse_loss(image_features1, image_features2)

        return loss
