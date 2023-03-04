##Date Modified: 3/4/23
##Author: Wolfgang Black

##The goal of this work is to try to determine how many Capuchin Birds are in a specific area of a forest
##To answer this a classifier will be built to recognize a capuchin call from an audio recording. 
##This model_dev.py scripts builds a CNN single-class classifier with either a spectrogram or mels-spectrogram preprocessing step.
##to change the type of sceptrogram, change the variable, spec_type in the variables box below. 
##
##Note: suggested usage is 'python model_dev.py' 
## unlike the model_dev in ../eda this uses the config file to generate the proper model in the case it doesn't already exist
#-----------------------------


##Import Libraries and functions
import os
import json
import tensorflow as tf

from utils.utils import (get_train_test_split,
    build_and_compile_CNN_model,
    get_model_performance_subplots,
    get_model_pred_and_reports)

#-----------------------------

#Read in json
with open('configs/config.json', 'r') as jsonfile:
    config = json.load(jsonfile)

pos_dir = config['pos_dir']
neg_dir = config['neg_dir']

#this variable determines the type of preprocessing the data undergoes before being formed into a dataset. see project readme for more of a description
spec_type = config['spec_type']

buffer = config['buffer']
batch = config['batch']
split = config['split'] #test split for the data
training_logit_threshold = config['training_logit_threshold']
epochs = config['epochs']
model_save_dir = config['artifacts_path']+'models/'

subplots_save_name = str(config['epochs'])+'_epochs_'+config['spec_type']+'_model_performance_subplots'
con_mat_plot_name = str(config['epochs'])+'_epochs_'+config['spec_type']+'_confusion_matrix'
class_rep_plot_name = str(config['epochs'])+'_epochs_'+config['spec_type']+'_classification_report'

##Below we'll run the code to train, validate, and save the model and corresponding figures

##Get the testing/training data
train, test = get_train_test_split(pos_dir,
                    neg_dir,
                    spec_type,
                    buffer,
                    batch,
                    split)

##Build a model, without specifying an optimizer, loss function, or metrics this will use Adam, BinaryCrossEntropy, 
##and report accuracy, recall, loss, and validation respectively

model = build_and_compile_CNN_model(train)

##Train model
hist = model.fit(train,
                 epochs = epochs,
                 validation_data = test,
                 callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=4, restore_best_weights = True))

##Save off training/validation performance metrics
get_model_performance_subplots(hist, subplots_save_name)

get_model_pred_and_reports(test, model, training_logit_threshold,  con_mat_plot_name, class_rep_plot_name)

##Save the model
try:
    os.listdir(model_save_dir)
except FileNotFoundError:
    os.mkdir(model_save_dir)    

model.save(model_save_dir+spec_type+'_model.h5')