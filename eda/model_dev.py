##Date Modified: 3/3/23
##Author: Wolfgang Black

##The goal of this work is to try to determine how many Capuchin Birds are in a specific area of a forest
##To answer this a classifier will be built to recognize a capuchin call from an audio recording. 
##This model_dev.py scripts builds a CNN single-class classifier with either a spectrogram or mels-spectrogram preprocessing step.
##to change the type of sceptrogram, change the variable, spec_type in the variables box below. 
##
##Note: suggested usage is 'python model_dev.py' 
#-----------------------------


##Import Libraries and functions
import os
import tensorflow as tf

from utils.utils import (get_train_test_split,
    build_and_compile_CNN_model,
    get_model_performance_subplots,
    get_model_pred_and_reports)

#-----------------------------

#Set some variables
pos_dir = '../data/Parsed_Capuchinbird_Clips/'
neg_dir = '../data/Parsed_Not_Capuchinbird_Clips/'

#this variable determines the type of preprocessing the data undergoes before being formed into a dataset. see project readme for more of a description
spec_type = 'spectrogram'
#spec_type = 'mels_spectrogram' 

buffer = 1000
batch = 16
split = 0.7 #test split for the data

epochs = 5
model_save_dir = '../artifacts/models/'
#-----------------------------
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
get_model_performance_subplots(hist, '5_epoch_spectrogram_model_performance_subplots')

get_model_pred_and_reports(test, model, 0.5,  '5_epoch_spectrogram_confusion_matrix','5_epoch_spectrogram_classification_report')

##Save the model
try:
    os.listdir(model_save_dir)
except FileNotFoundError:
    os.mkdir(model_save_dir)    

model.save_weights(model_save_dir+spec_type+'_model')