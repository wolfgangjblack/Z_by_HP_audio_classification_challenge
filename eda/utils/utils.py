import os
import tensorflow_io as tfio
import tensorflow as tf
from itertools import groupby

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from sklearn.metrics import confusion_matrix, classification_report


def load_wav_output_mono_channel_file(filename, sample_rate_out = 16000):
    """This function takes a filename, which is the full
     path of a specific .wav file, then decodes that file 
     to find the tensor associated with the sound - this is
     later used to get the spectrograms
    """
    #load encoded wav file
    file_contents = tf.io.read_file(filename)
    
    #Decode wav (tensors by channel)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels = 1)

    #Remove trailing axis
    wav = tf.squeeze(wav, axis = -1)
    sample_rate = tf.cast(sample_rate, dtype = tf.int64)

    #Goes from 44100Hz to 16000Hz - amplitude of the audio signal 
    wav = tfio.audio.resample(wav, rate_in = sample_rate, rate_out = sample_rate_out)

    return wav

def check_artifacts_dir(save_path = '../artifacts/'):
    '''This function checks if there is an artifacts dir in our roots 
    '''
    try:
         os.listdir(save_path)
    except FileNotFoundError:
        os.mkdir(save_path)
    return 

def plot_ex_wavs(wav, nwav, filename = 'example_wavelengths', save_path = '../artifacts/'):
    """This is meant to be a program which saves a comparison 
    graph of a capuchin and not a capuchin wav file plotted with a legend as a .png"""
    check_artifacts_dir(save_path)

    fig = plt.figure(figsize = (10, 10))
    plt.title("sample wav forms from capuchin and not a capuchin files")
    plt.plot(wav)
    plt.plot(nwav)
    fig.savefig(save_path+filename+'.png' )
    plt.close()
    return

def get_max_frames_from_mean_length(data_dir, sample_rate = 16000):
    lengths = []
    for file in os.listdir(data_dir):
        tensor_wave = load_wav_output_mono_channel_file(data_dir+file, sample_rate)
        lengths.append(len(tensor_wave))

    avg = tf.math.reduce_mean(lengths).numpy()
    min_time = round(avg/sample_rate)

    return sample_rate*min_time

def save_file_lengths_plot(data_dir, sample_rate = 16000, filename = 'training_wave_lengths', save_path = '../artifacts/'):
    '''This function reads in the data located in data_dir and plots the lengths of the files.
    This length is going to be the number of frames in the file, with the assumption the files 
    are sampled at 16k frames per second. This is then saved 
    '''

    check_artifacts_dir(save_path)
    lengths = []
    for file in os.listdir(data_dir):
        tensor_wave = load_wav_output_mono_channel_file(data_dir+file)
        lengths.append(len(tensor_wave))

    fig = plt.figure(figsize = (10, 10))
    plt.title("length plots with min, mean, and max lines")
    plt.plot(lengths, 'k--')
    plt.plot([i for i in range(len(lengths))], np.ones(len(lengths))*tf.math.reduce_mean(lengths).numpy(), 'r-')
    plt.plot([i for i in range(len(lengths))], np.ones(len(lengths))*tf.math.reduce_max(lengths).numpy(), 'b')
    plt.plot([i for i in range(len(lengths))], np.ones(len(lengths))*tf.math.reduce_min(lengths).numpy(), 'g')
    fig.savefig(save_path+filename+'.png')
    plt.close()
    return 


def preprocess(file_path:str, label:int, frames = 48000, sample_rate_out = 16000):
    '''This function reads in a .wav file path and its corresponding label (as an int)
     and then utilizes the load_wav_16k_mono function to read in the .wav and generate
      a spectrogram that can be used as an input into a CNN. 

      Note: frames is used here to indicate a max length of a recording. For instance,
       we assume our sample rate is 16000 frames/second, thus this 48000 allows for 3
        second recordings. 
    '''
    wav = load_wav_output_mono_channel_file(file_path, sample_rate_out)
    
    ##Select as much wav as fills frames, if len(wav) < frames, this will be less than frames and will need padding
    wav = wav[:frames]

    ##Calculate the number of zeros for padding, note if the wav >= frames, this will be empty
    
    zero_padding = tf.zeros([frames] - tf.shape(wav), dtype = tf.float32)

    ##Add zeros at the start IF the wav length < frames
    wav = tf.concat([zero_padding, wav], 0)

    #use short time fourier transform
    spectrogram = tf.signal.stft(wav, frame_length = 320, frame_step = 32)

    #Get the magnitude of the signal (los direction)
    spectrogram = tf.abs(spectrogram)

    #Adds a second dimension 
    spectrogram = tf.expand_dims(spectrogram, axis = 2)

    return spectrogram, label

def power_to_db(S, amin=1e-16, top_db=80.0):
    """Convert a power-spectrogram (magnitude squared) to decibel (dB) units.
    Computes the scaling ``10 * log10(S / max(S))`` in a numerically
    stable way.
    Based on:
    https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
    """
    def _tf_log10(x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator
    
    # Scale magnitude relative to maximum value in S. Zeros in the output 
    # correspond to positions where S == ref.
    ref = tf.reduce_max(S)

    log_spec = 10.0 * _tf_log10(tf.maximum(amin, S))
    log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref))

    log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

    return log_spec

def preprocess_to_mel_spectrogram(file_path, label, frames = 48000, ):
    wav = load_wav_output_mono_channel_file(file_path)
    
    ##Select as much wav as fills frames, if len(wav) < frames, this will be less than frames and will need padding
    wav = wav[:frames]

    ##Calculate the number of zeros for padding, note if the wav >= frames, this will be empty
    
    zero_padding = tf.zeros([frames] - tf.shape(wav), dtype = tf.float32)

    ##Add zeros at the start IF the wav length < frames
    wav = tf.concat([zero_padding, wav], 0)

    #use short time fourier transform
    spectrogram = tf.signal.stft(wav,
     frame_length = 320, ##This is fft_size
     frame_step = 32 ## this is hop_size
        ) #

    #Get the magnitude of the signal (los direction)
    spectrogram = tf.abs(spectrogram)
    
    mel_filter = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=100,
        num_spectrogram_bins = 257,
        sample_rate=frames,
        lower_edge_hertz=frames/100,
        upper_edge_hertz=frames/2,
        dtype=tf.dtypes.float32)

    mel_power_spectrogram = tf.matmul(tf.square(spectrogram), mel_filter)

    log_magnitude_mel_spectrograms = power_to_db(mel_power_spectrogram)

    log_magnitude_mel_spectrograms = tf.expand_dims(log_magnitude_mel_spectrograms, axis = 2)

    return log_magnitude_mel_spectrograms, label

def plot_spectrogram_subplot(pos_dir: str, neg_dir: str, filename = 'positive_negative_spectrogram_examples', save_path = '../artifacts/'):
    """This uses both the function preprocess and preprocess_to_mel_spectrogram to get spectrograms
    and mel spectrograms of the positive and negative classes and plot them in a single image. This image
    is then saved to artifacts.
    
    inputs:
        pos_dir: the directory to the positive class, here we expect this to be run from either src or eda files, thus we expect the data dir to be in the same master dir
            - expected value: '../data/Parsed_Capuchinbird_Clips/'
        neg_dir: the directory to the negative class, here we expect this to be run from either src or eda files, thus we expect the data dir to be in the same master dir
            - expected value: '../data/Parsed_Not_Capuchinbird_Clips/'
        filename: the file name that this file will be saved as, please note: this can be overwritten if not specified every time
        save_path: the artifacts directory

    Note: We've had some difficulty getting this to display evenly due to the size difference of the spectrograms vs the mels-grams. 
        Can likely change this with a gridspec_ arg, but this is so far untests
    """

    ##Check if artifacts exist
    check_artifacts_dir(save_path)

    ##Use pos and neg dirs to get data
    pos_dir = os.path.join('../data','Parsed_Capuchinbird_Clips/')
    neg_dir = '../data/Parsed_Not_Capuchinbird_Clips/'

    ##Get some data, along with labels
    pos = tf.data.Dataset.list_files(pos_dir+'*.wav')
    neg = tf.data.Dataset.list_files(neg_dir+'*.wav')

    positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
    negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))

    ##Get pos spectro and mel spectrograms
    filepath_pos, label_pos = positives.shuffle(buffer_size = 1000).as_numpy_iterator().next()
    spectrogram_pos, label_pos = preprocess(filepath_pos, label_pos)
    log_mag_mel_spec_pos, label_pos = preprocess_to_mel_spectrogram(filepath_pos, label_pos)

    ##Get neg spectro and mel spectrograms
    filepath_neg, label_neg = negatives.shuffle(buffer_size = 1000).as_numpy_iterator().next()
    spectrogram_neg, label_neg = preprocess(filepath_neg, label_neg)
    log_mag_mel_spec_neg, label_neg = preprocess_to_mel_spectrogram(filepath_neg, label_neg)

    ##Finally, build and save a subplot first showing pos/neg spectrograms, then pos/neg mels-grams
    fig, axs = plt.subplots(nrows = 4, ncols = 1, figsize = (20,10))
    axs[0].title.set_text('Positive Class Spectrogram')
    axs[0].imshow(tf.transpose(spectrogram_pos)[0])
    axs[1].title.set_text('Negative Class Spectrogram')
    axs[1].imshow(tf.transpose(spectrogram_neg)[0])
    axs[2].title.set_text('Positive Class Mels Spectrogram')
    axs[2].imshow(tf.transpose(log_mag_mel_spec_pos)[0])
    axs[3].title.set_text('Negative Class Mels Spectrogram')
    axs[3].imshow(tf.transpose(log_mag_mel_spec_neg)[0])
    plt.tight_layout
    fig.savefig(save_path+filename+'.png')
    return

def get_dataset(pos_dir: str, neg_dir: str):

     ##Use pos and neg dirs to get data
    pos_dir = os.path.join('../data','Parsed_Capuchinbird_Clips/')
    neg_dir = '../data/Parsed_Not_Capuchinbird_Clips/'

    ##Get some data, along with labels
    pos = tf.data.Dataset.list_files(pos_dir+'*.wav')
    neg = tf.data.Dataset.list_files(neg_dir+'*.wav')

    positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
    negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
    data = positives.concatenate(negatives)

    return data


def map_dataset_to_spectrograms(pos_dir:str, neg_dir:str, spec_type:str, buffer:int, batch_size:int):
    """This function creates a data structure that is usable by tensorflow. It takes our dataset, 
    formed by get_data and consisting of labels and filepaths, and then uses a mapping with the specified
    spectrogram type, as denoted by spec_type. Finally, we shuffle the data, choose a batch and buffer size,
    and then return the dataset. 
    """
    data = get_dataset(pos_dir, neg_dir)
    if spec_type == 'spectrogram':
        data = data.map(preprocess)
    elif spec_type == 'mels_spectrogram':
        data = data.map(preprocess_to_mel_spectrogram)

    data = data.cache()
    data = data.shuffle(buffer_size = buffer)
    data = data.batch(batch_size)
    data = data.prefetch(int(batch_size/2))
    
    return data

def get_test_train_split(pos_dir:str, neg_dir:str, spec_type:str, buffer:int, batch_size:int, split_size:float):
    """This function uses map_dataset_to_spectrograms to transform a directory of wav files into spectrograms,
    type denoted by spec_type, and then splits the data into a train and test dataset using the split_size. split_size
    is a float, meant to be between 0-1, and determines the size of the train set, with the remainder used for test"""
    data = map_dataset_to_spectrograms(pos_dir, neg_dir, spec_type, buffer, batch_size)
    split = int(len(data)*split_size)+1

    train = data.take(split)
    test = data.skip(split).take(len(data)-split)

    return train, test

def build_and_compile_CNN_model(train,
            optimizer = 'Adam',
            loss_func = 'BinaryCrossentropy',
            metric_arg_list =[tf.keras.metrics.Accuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]):

    samples, _ = train.as_numpy_iterator().next()
    input_shape = samples.shape[1:]

    model = Sequential()
    model.add(Conv2D(16, (3,3), activation = 'relu', input_shape = input_shape)) ##This is 16 kernels of shape 3x3, input shape 
    model.add(Conv2D(16, (3,3), activation = 'relu')) ## as this is the Second layer, we no longer need an input shape - its connected directly to the prior layer
    model.add(MaxPooling2D()),
    model.add(Dropout(0.25)),
    model.add(Flatten()) #This combines all the nodes from the previous conv2d layer into a single dimension
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(optimizer, 
                loss = loss_func,
                metrics = metric_arg_list)
    return model

def get_model_performance_subplots(model, filename = 'model_performance_subplots', save_path =  '../artifacts/'):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (10, 10))

    ax1.title.set_text("Accuracy")
    ax1.plot(model.history['accuracy'], 'r')
    ax1.plot(model.history['val_accuracy'], 'b')
    ax1.legend(['train', 'val'])

    ax2.title.set_text("Loss")
    ax2.plot(model.history['loss'], 'r')
    ax2.plot(model.history['val_loss'], 'b')
    ax2.set_ylim([0, 1])

    ax3.title.set_text("Recall")
    ax3.plot(model.history['recall_1'], 'r')
    ax3.plot(model.history['val_recall_1'], 'b')

    ax4.title.set_text("Precision")
    ax4.plot(model.history['precision_1'], 'r')
    ax4.plot(model.history['val_precision_1'], 'b')

    fig.savefig(save_path+filename+'.png')
    plt.close()
    return

def get_model_pred_and_reports(test_data,
        model,
        logit_threshold = 0.5, 
        confusion_matrix_filename = 'confusion_matrix',
        classification_report_filename = 'classification_report', save_path = '../artifacts/'):
    
    X_test, y_test = test_data.as_numpy_iterator().next()

    yhat = model.predict(X_test)

    ##Convert logits to classes
    yhat = [1 if prediction > logit_threshold else 0 for prediction in yhat]

    save_classification_report(y_test, yhat, classification_report_filename, save_path)
    save_confusion_matrix(y_test, yhat, confusion_matrix_filename, save_path)
    return


def save_classification_report(y_true, yhat, filename = 'classification_report', save_path = '../artifacts/'):
    textfile = open(save_path+filename+'.txt', 'w')
    textfile.write(classification_report(y_true, yhat) )
    textfile.close()
    return

def save_confusion_matrix(y_true, yhat, filename = 'confusion_matrix', save_path =  '../artifacts/'):
    
    confusion_mtx = tf.math.confusion_matrix(y_true, yhat)
    plt.figure(figsize=(10, 8))
    fig = sns.heatmap(confusion_mtx,
                xticklabels=[0,1],
                yticklabels=[0,1],
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show() 
    fig.savefig()
    plt.close()
    return

def load_mp3_output_mono_channel_file(filename, sample_rate_out = 16000):
  """ Load an mp3 file, convert it to a float tensor, resample
  to some specified output sample rate
  """
  res = tfio.audio.AudioIOTensor(filename)
  # Convert to tensor and combine channels
  tensor = res.to_tensor()
  tensor = tf.math.reduce_sum(tensor, axis = 1) / 2
  #extract sample rate and cast to tf.datatype
  sample_rate = res.rate
  sample_rate = tf.cast(sample_rate, dtype = tf.int64)
  # Resample to sample_rate_out
  wav = tfio.audio.resample(tensor, rate_in = sample_rate, rate_out = sample_rate_out)
  return wav

def preprocess_mp3(sample):
    """This is for inference, not necessarily for training. As such, this does not produce a label like previouspreprocessing  functions"""
    sample = sample[0]
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

def preprocess_to_mel_spectrogram_mp3(sample, frames = 48000):
    """This is for inference, not necessarily for training. As such, this does not produce a label like previouspreprocessing  functions"""

    ##Calculate the number of zeros for padding, note if the wav >= frames, this will be empty
    sample = sample[0]
    zero_padding = tf.zeros([frames] - tf.shape(sample), dtype = tf.float32)

    ##Add zeros at the start IF the wav length < frames
    wav = tf.concat([zero_padding, sample], 0)

    #use short time fourier transform
    spectrogram = tf.signal.stft(wav,
     frame_length = 320, ##This is fft_size
     frame_step = 32 ## this is hop_size
        ) #

    #Get the magnitude of the signal (los direction)
    spectrogram = tf.abs(spectrogram)
    
    mel_filter = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=100,
        num_spectrogram_bins = 257,
        sample_rate=frames,
        lower_edge_hertz=frames/100,
        upper_edge_hertz=frames/2,
        dtype=tf.dtypes.float32)

    mel_power_spectrogram = tf.matmul(tf.square(spectrogram), mel_filter)

    log_magnitude_mel_spectrograms = power_to_db(mel_power_spectrogram)

    log_magnitude_mel_spectrograms = tf.expand_dims(log_magnitude_mel_spectrograms, axis = 2)

    return log_magnitude_mel_spectrograms

def forest_recordings_inference(spec_type:str, 
            inf_batch_size: int,
            model,
            filename = 'results',
            data_path = '../data/Forest Recordings/',
            frames = 48000,
            save_path = '../artifacts/'):
    """ This function takes files that are MUCH longer in a different directory and 
    attempts to find whether or not there is a positive class in the files. As such
    it has slightly different preprocessing functions it utilizes. While most of
    this could be abstracted out of the function and put into the main .py. However, 
    to do that we'd need to output the file variable. As such, I just put this into a big function...
    """

    results = {}
    ## Process files from the forest_recordings for data, 
    for file in os.listdir():
        FILEPATH = os.path.join(data_path, file)
        
        wav = load_mp3_output_mono_channel_file(FILEPATH)

        audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav,
                 wav,
                sequence_length = frames,
                sequence_stride = frames,
                batch_size = inf_batch_size)

        if spec_type == 'spectrogram':
            audio_slices = audio_slices.map(preprocess_mp3)
        elif spec_type == 'mels_spectrogram':
            audio_slices = audio_slices.map(preprocess_to_mel_spectrogram_mp3)

        audio_slices = audio_slices.batch(inf_batch_size)
        
        yhat = model.predict(audio_slices)
        
        results[file] = yhat

    class_preds = {}

    for file, logits in results.items():

        class_preds[file] = [1 if prediction > 0.99 else 0 for prediction in logits]

    postprocessed = {}
    
    for file, scores in class_preds.items():

        postprocessed[file] = tf.math.reduce_sum([key for key, group in groupby(scores)]).numpy()

    postprocessed

    with open(save_path + filename+'.csv', 'w', newline='') as f:

        writer = csv.writer(f, delimiter=',')

        writer.writerow(['recording', 'capuchin_calls'])

        for key, value in postprocessed.items():

            writer.writerow([key, value])
    return