import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from sign.dataset.university import U1652DatasetEval, get_transforms
from sign.evaluate.university import evaluate
from sign.model import TimmModel


@dataclass
class Configuration:

    # Model
    # model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    model: str = 'vit_large_patch14_dinov2.lvd142m'
    
    # Override model image size
    img_size: int = 518
    
    # Evaluation
    batch_size: int = 128
    verbose: bool = True
    gpu_ids: tuple = (5,)
    normalize_features: bool = True
    eval_gallery_n: int = -1             # -1 for all or int
    
    # Dataset
    dataset: str = 'S2D'           # 'U1652-D2S' | 'U1652-S2D'
    
    # Checkpoint to start from
    checkpoint_start = 'sues200/vit_large_patch14_dinov2.lvd142m/250/weights_e1_0.9870.pth'
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda:5' if torch.cuda.is_available() else 'cpu' 

    handcraft_model: bool = True
    views: int = 2
    nclasses: int = 200
    block: int = 3
    triplet_loss: float = 0.3
    resnet: bool = False

    # sues200
    altitude: int = 250 # "150|200|250|300"
    

#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration() 

if config.dataset == 'D2S':
    config.query_folder_train = f'/home3/mzq/data/SUES-200-512x512/train/{config.altitude}/satellite'
    config.gallery_folder_train = f'/home3/mzq/data/SUES-200-512x512/train/{config.altitude}/drone'
    config.query_folder_test = f'/home3/mzq/data/SUES-200-512x512/test/{config.altitude}/query_drone'
    config.gallery_folder_test = f'/home3/mzq/data/SUES-200-512x512/test/{config.altitude}/gallery_satellite'
elif config.dataset == 'S2D':
    config.query_folder_train = f'/home3/mzq/data/SUES-200-512x512/train/{config.altitude}/satellite'
    config.gallery_folder_train = f'/home3/mzq/data/SUES-200-512x512/train/{config.altitude}/drone'
    config.query_folder_test = f'/home3/mzq/data/SUES-200-512x512/test/{config.altitude}/query_satellite'
    config.gallery_folder_test = f'/home3/mzq/data/SUES-200-512x512/test/{config.altitude}/gallery_drone'

if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))


    if config.handcraft_model is not True:
        print("\nModel: {}".format(config.model))
        model = TimmModel(config.model,
                          pretrained=True,
                          img_size=config.img_size)
    # elif "convnext" in config.model:
    #     from sample4geo.hand_convnext.model import make_model
    #     model = make_model(config)
    #     print("\nModel:{}".format("adjust model: handcraft convnext-base"))
    elif "dinov2" in config.model:
        model = TimmModel(config)
        print("\nModel:{}".format("adjust model: handcraft dinov2"))
    elif "eva02" in config.model:
        model = TimmModel(config)
        print("\nModel:{}".format("adjust model: handcraft eva02"))
                          
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)
    

    # load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            
    # Model to device   
    model = model.to(config.device)

    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)
                                                                                                                                 
    
    # Reference Satellite Images
    query_dataset_test = U1652DatasetEval(data_folder=config.query_folder_test,
                                               mode="query",
                                               transforms=val_transforms,
                                               )
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    # Query Ground Images Test
    gallery_dataset_test = U1652DatasetEval(data_folder=config.gallery_folder_test,
                                               mode="gallery",
                                               transforms=val_transforms,
                                               sample_ids=query_dataset_test.get_sample_ids(),
                                               gallery_n=config.eval_gallery_n,
                                               )
    
    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))
   

    print("\n{}[{}]{}".format(30*"-", "SUES-200", 30*"-"))  

    r1_test = evaluate(config=config,
                       model=model,
                       query_loader=query_dataloader_test,
                       gallery_loader=gallery_dataloader_test, 
                       ranks=[1, 5, 10],
                       step_size=1000,
                       cleanup=True)
 
