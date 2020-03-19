#! /usr/bin/env python
import argparse
import os
import numpy as np
import json
from fastai.vision import *
from fastai import *
from pathlib import Path

def create_training_data(
    train_image_folder,
    validate_folder_name,
    test_image_folder,
    labels,
    gridsize,
    batchsize
):
    data=processData(train_image_folder,validate_folder_name,gridsize,batchsize)
    return data


def create_transfer_model(
    model_folder,
    model_name,
    gpu,
    lr
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if os.path.exists(model_folder):
        learn = load_learner(model_folder,model_name)
    return learn  
    
def create_model(
    data,
    gpu,
    lr
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    learn = cnn_learner(data,models.resnet34,metrics=[accuracy,Precision(),Recall(),FBeta()])
    return learn        


def train_transfer_model(
    learn,
    data,
    save_model_path,
    save_model_name

):
    learn.data=data
    learn.fit(1)
    learn.export(save_model_path,save_model_name)

def train_model(
    learn,
    data,
    save_model_path,
    save_model_name

):
    learn.data=data
    lr = 3e-3
    learn.fit_one_cycle(3,slice(lr))
    learn.unfreeze()
    learn.fit_one_cycle(5,slice(1e-5,lr/5))
    learn.export(save_model_path,save_model_name)   
    

#If validation folder is present it should be a sibling of training folder.
#Only pass the name of validation folder 
def processData(train_folder,validate_folder_name,imagesize,batchsize):
    tfms = get_transforms()
    train_path = Path(train_folder)
    if validate_folder_name =="":
      return (ImageList
              .from_folder(train_folder)
              .split_by_rand_pct()
              .label_from_folder()
              .transform(tfms,size=(imagesize,imagesize))
              .databunch(bs=batchsize)
              .normalize(imagenet_stats)
      )
    else:
      return (ImageList.from_folder(train_path.parent)
              .split_by_folder(train='train', valid=validate_folder_name)
              .label_from_folder()
              .transform(tfms,size=(imagesize,imagesize))
              .databunch(bs=batchsize)
              .normalize(imagenet_stats)
      )



def _main_(args):
    config_path  = args.conf
    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)    
    #   Parse the annotations
    ###############################
    data = create_training_data(
        config['train']['train_image_folder'],
        config['train']['validation_folder_name'],
        config['train']['test_image_folder'],
        config['train']['labels'],
        config['train']['grid_size'],
        config['train']['batch_size']  
    )
    print(data)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    if(config['train']['use_transfer_model'] == True):
        learn = create_transfer_model(
            config['model']['transfer_model_folder'],
            config['model']['transfer_model_name'],
            config['train']['gpu'],
            config['train']['learning_rate']
        )
        train_transfer_model(
            learn,
            data,
            save_model_path,
            save_model_name
        )    
    else:
        learn = create_model(
            data,
            config['train']['gpu'],
            config['train']['learning_rate']
        )
        train_model(
            learn,
            data,
            config['model']['save_model_folder'],
            config['model']['save_model_name']  
        )    

    

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
    description='Train classification model')
    argparser.add_argument(
        '-c', '--conf', help='path to configuration file')
    args = argparser.parse_args()
    _main_(args)
