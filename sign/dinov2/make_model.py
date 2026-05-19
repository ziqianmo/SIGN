import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from timm.models import create_model
import numpy as np
from torch.nn import init
from torch.nn.parameter import Parameter
from torch import cosine_similarity
import os
from sign.Utils import init
import math
import timm

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.training:
            if self.return_f:
                f = x
                x = self.classifier(x)
                return x, f
            else:
                x = self.classifier(x)
                return x
        else:
            return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)
    



class build_dinov2(nn.Module):
    def __init__(self, config, num_classes, block=4, return_f=False, resnet=False):
        super(build_dinov2, self).__init__()
        self.return_f = return_f
        self.in_planes = 1024 #large1024,base768,small384

        self.model = timm.create_model(config.model, pretrained=True, num_classes=0, pretrained_cfg_overlay=dict(file=r'/home/mzq/.cache/huggingface/hub/models--timm--vit_large_patch14_dinov2.lvd142m/pytorch_model.bin'))
        self.model = self.model.eval()

        self.num_classes = num_classes
        self.classifier1 = ClassBlock(self.in_planes, num_classes, 0.5, return_f=return_f)
        self.block = block

        for i in range(self.block):
            name = 'classifier_heat' + str(i + 1)
            setattr(self, name, ClassBlock(self.in_planes, num_classes, 0.5, return_f=self.return_f))



    def forward(self, x):

        # 1. 获取 patch 嵌入
        vit_features = self.model.patch_embed(x)  # 转换为嵌入特征 [batch_size, c, embed_dim]

        # 2. 添加分类 token
        cls_token = self.model.cls_token.expand(vit_features.size(0), -1, -1)  # [batch_size, 1, embed_dim]
        vit_features = torch.cat((cls_token, vit_features), dim=1)  # [batch_size, c+1, embed_dim]

        # 3. 添加位置嵌入
        vit_features = vit_features + self.model.pos_embed  # 匹配位置嵌入


        # 2. 冻结前几个块
        with torch.no_grad():
            for blk in self.model.blocks[:-2]:  # 前 N-2 块冻结
                vit_features = blk(vit_features)
        vit_features = vit_features.detach()  # 分离冻结部分的计算图

        # 3. 训练后两个块
        for blk in self.model.blocks[-2:]:  # 后 2 块参与训练
            vit_features = blk(vit_features)

        
        # vit_features = self.model.forward_features(x)
        cls_feature = vit_features[:,0]
        part_features = vit_features[:,1:]
        gap_features = part_features.mean([1])
        

        transformer_feature = self.classifier1(cls_feature)
        heat_result = self.get_heatmap_pool(part_features)

        y = self.part_classifier(self.block, heat_result, cls_name='classifier_heat')


        # -- Training
        if self.training:
            y = y + [transformer_feature]
            if self.return_f:  # return_f是triplet loss的设置，0.3
                cls, features = [], []
                for i in y:
                    cls.append(i[0])
                    features.append(i[1])
                return cls, features, heat_result, gap_features

        # -- Eval
        else:
            # transformer_feature = transformer_feature.view(transformer_feature.size(0),-1,1)
            # y = torch.cat([y, transformer_feature],dim=2)
            pass
             

        # return y
        return gap_features
#       return gap_features, part_features
    
    def get_heatmap_pool(self, part_features, add_global=False, otherbranch=False):
        heatmap = torch.mean(part_features,dim=-1)
        size = part_features.size(1)
        arg = torch.argsort(heatmap, dim=1, descending=True)
        x_sort = [part_features[i, arg[i], :] for i in range(part_features.size(0))]
        x_sort = torch.stack(x_sort, dim=0)

        split_each = size / self.block
        split_list = [int(split_each) for i in range(self.block - 1)]
        split_list.append(size - sum(split_list))
        split_x = x_sort.split(split_list, dim=1)

        split_list = [torch.mean(split, dim=1) for split in split_x]
        part_featuers_ = torch.stack(split_list, dim=2)
        if add_global:
            global_feat = torch.mean(part_features, dim=1).view(part_features.size(0), -1, 1).expand(-1, -1, self.block)
            part_featuers_ = part_featuers_ + global_feat
        if otherbranch:
            otherbranch_ = torch.mean(torch.stack(split_list[1:], dim=2), dim=-1)
            return part_featuers_, otherbranch_
        return part_featuers_

    def part_classifier(self, block, x, cls_name='classifier_heat'):
        part = {}
        predict = {}
        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            name = cls_name + str(i + 1)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y




def make_dinov2_model(config, num_class, block=4, return_f=False, resnet=False):
    print('===========building dinov2===========')
    model = build_dinov2(config, num_class, block=block, return_f=return_f, resnet=resnet)
    return model
