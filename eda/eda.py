##Date Modified: 3/3/23
##Author: Wolfgang Black

##This script is meant to perform an EDA on the data sources for an audio classification problem
##The goal of this work is to try to determine how many Capuchin Birds are in a specific area of a forest
##To answer this a classifier will be built to recognize a capuchin call from an audio recording. 

##Import Libraries and functions
import os

import tensorflow as tf
import tensorflow_io as tfio

from utils.utils import (load_wav_output_mono_channel_file,
    plot_ex_wavs,
    save_file_lengths_plot,
    plot_spectrogram_subplot)


##looking at a specified file in our positive/negative class data directories, plot and save figures with
##some basic data understanding. 

#Set some directories
pos_dir = '../data/Parsed_Capuchinbird_Clips/'
neg_dir = '../data/Parsed_Not_Capuchinbird_Clips/'

capuchin_file = pos_dir + 'XC3776-3.wav'
not_capuchin_file = neg_dir + 'afternoon-birds-song-in-forest-0.wav'

wav = load_wav_output_mono_channel_file(capuchin_file)
nwav =load_wav_output_mono_channel_file(not_capuchin_file)

##Plot and save figures. Figures include wave forms for different calls
##Audio lengths, and two different types of spectrograms. More about
##Spectrograms can be read in the readme in the github.

plot_ex_wavs(wav, nwav)

save_file_lengths_plot(pos_dir, 16000)

plot_spectrogram_subplot(pos_dir, neg_dir)
