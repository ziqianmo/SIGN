import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.amp import autocast
import torch.nn.functional as F
from sign.loss.cal_loss import cal_kl_loss, cal_loss1, cal_loss2, cal_triplet_loss
from sign.loss.triplet_loss import Tripletloss
import torch.nn as nn
import numpy as np

def train(train_config, model, dataloader, loss_functions, optimizer, scheduler=None, scaler=None):

    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    criterion = nn.CrossEntropyLoss()
    # triplet_loss = Tripletloss(margin=train_config.triplet_loss)
    # split_num = train_config.batch_size//train_config.sample_num

    # for loop over one epoch
    for query, reference, ids, labels in bar:
        
        if scaler:
            with autocast('cuda'):
            
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                labels = labels.to(train_config.device)

                if train_config.handcraft_model is not True:
                    features1, features2 = model(query, reference, dino=True)
                else:
                    output1, output2 = model(query, reference)
                    features_cls_1, features_cls_2 = output1[0], output2[0]  # -- for classifier
                    features_heat_1, features_heat_2 = output1[2], output2[2]
                    features_gap1, features_gap2 = output1[3], output2[3]

                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 

                    loss_cls = cal_loss1(features_cls_1, labels, criterion) + cal_loss2(features_cls_2, labels, criterion)


                    loss_D_S = loss_functions["blocks_mse"](features_heat_1, features_heat_2)
                                                                      

                    loss_D_D = loss_functions["blocks_infoNCE"](features_heat_1, features_heat_2,
                                                                          model.module.logit_scale_blocks.exp())
                    loss_S_S = loss_functions["blocks_infoNCE"](features_heat_1, features_heat_2,
                                                                          model.module.logit_scale_blocks.exp())

                else:
                    loss = loss_functions["infoNCE"](features_gap1, features_gap2,
                                                                          model.logit_scale_blocks.exp())
                    
                    loss_cls = cal_loss1(features_cls_1, labels, criterion) + cal_loss2(features_cls_2, labels, criterion)
                    

                    loss_D_S = loss_functions["blocks_infoNCE"](features_heat_1, features_heat_2,
                                                                          model.logit_scale_blocks.exp())
                    loss_D_D = loss_functions["blocks_infoNCE"](features_heat_1, features_heat_1,
                                                                          model.logit_scale_blocks.exp())
                    loss_S_S = loss_functions["blocks_infoNCE"](features_heat_2, features_heat_2,
                                                                          model.logit_scale_blocks.exp())
                    

                lossall = loss + loss_cls + loss_D_S + loss_D_D + loss_S_S

                losses.update(lossall.item())
                
                  
            scaler.scale(lossall).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
        else:
        
            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss = loss_functions["infoNCE"](features1, features2, model.module.logit_scale.exp()) 
            else:
                loss = loss_functions["infoNCE"](features1, features2, model.logit_scale.exp()) 
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        
        
        
        if train_config.verbose:
            
            monitor = {"loss_cls": "{:.4f}".format(loss_cls.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    
    ids_list = []
    with torch.no_grad():
        
        for img, ids in bar:
        
            ids_list.append(ids)
            
            with autocast('cuda'):
         
                img = img.to(train_config.device)

                if train_config.handcraft_model is not True:
                    img_feature = model(img)
                else:
                    output = model(img)
                    img_feature = output
                
            
                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
      
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return img_features, ids_list