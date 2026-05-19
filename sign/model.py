import torch
import timm
import numpy as np
import torch.nn as nn
from sign.dinov2.make_model import make_dinov2_model


class TimmModel(nn.Module):

    def __init__(self, config):  # 调整 img_size 为 384
                
        super(TimmModel, self).__init__()
        
        self.img_size = config.img_size
        
        self.model = make_dinov2_model(config, num_class=config.nclasses, block=config.block, return_f=config.triplet_loss, resnet=config.resnet)
        
        # 用于缩放 logits
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_blocks = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)
        
    def forward(self, x1, x2=None, dino=False):
        """
        query_img: 单张卫星图像
        gallery_imgs: 无人机图像序列
        """
        if x2 is not None:
            y1 = self.model(x1)
            y2 = self.model(x2)
            return y1, y2
        else:
            y1 = self.model(x1)
            return y1
