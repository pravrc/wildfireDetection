
# Early Wildfire Detection from Images

A project for detection and classification of wildfire images

This project has two repositories. One for classification and one for wildfire detection using bounding boxes.
Fire detection for yolov3 is using the forked version of https://github.com/experiencor/keras-yolo3 along with some modifications

---
## Setup and Installation:

### Prerequisites:
Python3

1. Clone the following repos
``` 
    git clone https://github.com/pravrc/wildfireDetection.git
    git clone https://github.com/pravrc/yolov3.git 
```
2. Install packages
```
    pip install -r requirements.txt
```
or 
```
    pip install fastai
    pip install tensorflow
    pip install opencv-contrib-python
```
    
---
## DATASETS
Datasets for training and testing can be downloaded from S3 using wget
```    
    wget http://pravrc-wildfire.s3.amazonaws.com/wildfiredata.zip
    unzip wildfiredata.zip
```   
wildfire_train_data and wildfire_test_data are the folders containing training and test data respectively

---
## TRAINING
### 1. Classification:
Make sure you are in ~/wildfireDetection. Make sure you setup all the training configuration
in file ~/wildfireDetection/configs/trainClassifier.json. The existing sample has preset values 
which can be edited
```        
    cd ~/wildfireDetection
    python wildfireDetection/trainClassifier.py -c configs/trainClassifier.json
```   
### 2.Detection:
Edit config.json appropriately
```       
    cd ~/keras-yolo3
    python train.py -c config.json
```
---
## INFERENCE
### 1. Classification 
Make sure you setup all the inference configuration 
in file ~/wildfireDetection/configs/inferenceClassifier.json. The existing sample has preset values 
which can be edited
```
    cd ~/wildfireDetection
    python wildfireDetection/inferenceClassifier.py -c configs/inferenceClassifier.json    
```
### 2. Detection
Edit config.json to set parameters
```
    cd ~/keras-yolo3
    python predict.py -c config.json -i INPUT_FILE
```   
---

