import os
import tensorflow_io as tfio
import tensorflow as tf
import matplotlib.pyplot as plt

def load_wav_16k_mono(filename):
    """This is a load wav function"""
    #load encoded wav file
    file_contents = tf.io.read_file(filename)
    
    #Decode wav (tensors by channel)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels = 1)

    #Remove trailing axis
    wav = tf.squeeze(wav, axis = -1)
    sample_rate = tf.cast(sample_rate, dtype = tf.int64)

    #Goes from 44100Hz to 16000Hz - amplitude of the audio signal 
    wav = tfio.audio.resample(wav, rate_in = sample_rate, rate_out = 16000)

    return wav

def plot_ex_wavs(wav, nwav, save_path = '../artifacts/'):
    """This is meant to be a program which saves a comparison 
    graph of a capuchin and not a capuchin wav file plotted with a legend as a .png"""
    check_artifacts_dir(save_path)

    fig = plt.figure(figsize = (10, 10))
    plt.title("test plot")
    plt.plot(wav)
    plt.plot(nwav)
    fig.savefig(save_path)
    plt.close()
    return

def check_artifacts_dir(save_path = '../artifacts/'):
    try:
         os.listdir(save_path)
    except FileNotFoundError:
        os.mkdir(save_path)
    return 


