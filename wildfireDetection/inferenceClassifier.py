#! /usr/bin/env python
import argparse
import os
import numpy as np
import json
from fastai import *
from fastai.vision import *


def predict(learn,input_path):
    image_paths = []
    if os.path.isdir(input_path):
        for inp_file in os.listdir(input_path):
            image_paths += [input_path + inp_file]
    else:
        image_paths += [input_path]
        image_paths = [inp_file for inp_file in image_paths if (
            inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]
    # the main loop
    for image_path in image_paths:
        #image = cv2.imread(image_path)
        image=open_image(image_path)
        print(image_path)
        print(learn.predict(image))


def _main_(args):
    config_path  = args.conf
    input_path   = args.input
    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)
    learn = load_learner(config['model']['folder'],config['model']['name'])

    # Do predictions for the images
    predict(learn,config['input']['images_folder'])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='classify as smoke or no smoke')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images') 
    args = argparser.parse_args()
    _main_(args)
