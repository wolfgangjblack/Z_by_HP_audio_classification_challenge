import os
import tensorflow as tf
import tensorflow_io as tfio

from utils.utils import load_wav_16k_mono, plot_ex_wavs

capuchin_file = '../data/Parsed_Capuchinbird_Clips/XC3776-3.wav'
not_capuchin_file = '../data/Parsed_Not_Capuchinbird_Clips/afternoon-birds-song-in-forest-0.wav'

print('about to call func')
wav = load_wav_16k_mono(capuchin_file)
nwav =load_wav_16k_mono(not_capuchin_file)


print('have loaded the wave through function')

plot_ex_wavs(wav, nwav)

print('have saved graph')
