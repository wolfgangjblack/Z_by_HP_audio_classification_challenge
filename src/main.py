##Date Modified: 3/4/23
##Author: Wolfgang Black

##The goal of this work is to try to determine how many Capuchin Birds are in a specific area of a forest
## To answer this a classifier will be built to recognize a capuchin call from an audio recording. 
## This main.py script checks to see if a model as been built and saved in a model_path directory
## if not, it builds a CNN single-class classifier with either a spectrogram or mels-spectrogram preprocessing step.
## After building a model, or if a model already exists, this will read in the unlabeled data (forest recordings), make
## a prediction, and then generates a csv in results_dir 
##
##Note: suggested usage is 'python main.py' 
#-----------------------------

##Import Libraries and functions
import os
import json
import subprocess

import tensorflow as tf
from tensorflow.keras.models import load_model

from utils.utils import (check_artifacts_dir,
    forest_recordings_inference)

#-----------------------------

with open('configs/config.json', 'r') as jsonfile:
    config = json.load(jsonfile)

artifacts_path = config['artifacts_path']
model_path = config['artifacts_path']+'models/'
spec_type = config['spec_type']
inf_batch_size = config['inf_batch_size']
results_filename = config['results_filename']
unlabeled_data_path = config['unlabeled_data_path']
frames = config['frames']
need_to_eda = config['need_to_eda']
need_to_train = config['need_to_train']
inference_logit_threshold = config['inference_logit_threshold']

try:
    os.listdir(model_path+spec_type+'_model.h5')
except FileNotFoundError: 
    print("spectrogram type model not found")
    need_to_train = 1

if need_to_train == 1:
    try:
        os.listdir(artifacts_path)
    except:
        print('no model directory, will run eda')
        need_to_eda = 1

    ##If eda is needed, we'll run the eda.py
    if need_to_eda == 1:
        print("running EDA")
        check_artifacts_dir(artifacts_path)
        subprocess.call('python eda.py', shell = True)
    
    print("training model")
    check_artifacts_dir(model_path)
    subprocess.call("python model_dev_from_config.py", shell = True)

print('loading model')
model = load_model(model_path+spec_type+'_model.h5')

print('generating results for unlabeled data')

forest_recordings_inference(spec_type, 
        inf_batch_size,
        model,
        spec_type+'_'+results_filename,
        unlabeled_data_path,
        frames,
        inference_logit_threshold,
        artifacts_path)

print('finished')