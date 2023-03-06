# Z_by_HP_audio_classification_challenge

Date Modified: 3/4/23 <br>
Author: Wolfgang Black <br>

## Task: 
As per the challenge on [kaggle](https://www.kaggle.com/datasets/kenjee/z-by-hp-unlocked-challenge-3-signal-processing), the task is to build an audio classifier which can identify a capuchin call. This classifier will then need to listen to recordings of various length and count up the number of capuchin calls per recording. 

## The Data:
Download [here](https://www.kaggle.com/datasets/kenjee/z-by-hp-unlocked-challenge-3-signal-processing). The data is seperated into training data and the final evaluation data. The training data contains files <i> Parsed_Capuchinbird_Clips</i> and <i> Parsed_Not_Capuchinbird_Clips </i>. The evaluation data, which is unlabeled but will be used in the contest to evaluate the model performance, is found in <i> Forest Recordings </i>.

## The model:

In this work, we used a Convolutional Nueral Network for binary image classification. Using a CNN for image classification involves training a deep learning model on a dataset of labeled images. The CNN model architecture consists of multiple layers, including convolutional layers, pooling layers, and fully connected layers. The convolutional layers use filters to extract features from the input images, while the pooling layers downsample the feature maps to reduce the number of parameters. The fully connected layers then use the extracted features to classify the images.

After the architecture is selected, the model is trained on the prepared dataset using backpropagation to adjust the weights of the network to minimize the classification error. This involves feeding the images through the network, calculating the loss, and updating the weights using an optimization algorithm like stochastic gradient descent.

Using a CNN for image classification has become a standard approach in computer vision and has been successfully applied in various applications, including object recognition, face recognition, and medical imaging.

Since CNNs are used for image classification, we need to process the audio data into something a CNN can recognize. Standard practice is to use spectrograms. 

### Why Spectrograms
Spectrograms are commonly used as inputs for audio deep learning because they provide a time-frequency representation of the audio signal that can be easily analyzed using machine learning algorithms.

A spectrogram is a visual representation of the spectrum of frequencies of a sound wave as it varies with time. It is created by taking a series of short time slices of the audio signal and calculating the frequency spectrum for each slice. The resulting spectrogram shows how the energy in different frequency bands changes over time.

Spectrograms are useful for audio analysis because they provide a compact representation of the time-varying spectral content of a sound. This allows machine learning models to learn features that are important for classification or regression tasks. For example, a deep learning model trained on spectrograms can learn to recognize different speech sounds, musical instruments, or environmental sounds.

Below we'll explore designing functions to turn the .wav files into either a standard spectrogram or a mels spectrogram as use for our inputs.

### Spectrograms vs Mels Spectrograms
Spectrograms and mel spectrograms are both time-frequency representations of audio signals, but they differ in how they represent frequency.

A spectrogram is a visual representation of the spectrum of frequencies of a sound wave as it varies with time. It is created by taking a series of short time slices of the audio signal and calculating the frequency spectrum for each slice. The resulting spectrogram shows how the energy in different frequency bands changes over time.

A mel spectrogram is a variant of the spectrogram that is obtained by applying a mel scale to the frequency axis. The mel scale is a perceptual scale of pitches that is based on how humans perceive sound. Mel spectrograms are useful because they allow the representation of audio signals in a way that is more aligned with human perception of sound.

In other words, while a standard spectrogram represents frequency linearly, a mel spectrogram represents frequency logarithmically, with greater resolution in the lower frequencies where humans are more sensitive to changes in pitch. This can be particularly useful for tasks such as speech recognition, where the important information is often concentrated in the lower frequencies.

## Usage
### EDA
To see the work done for EDA, a user can either use the notebook [code_and_data_exploratory_analysis](https://github.com/wolfgangjblack/Z_by_HP_audio_classification_challenge/blob/finalize_readme/eda/code_and_data_exploratory_analysis.ipynb) to see the code internal to the functions and interact with the data. Otherwise, the scripts eda.py and model_dev.py in the /eda/ directory will generate data and model artifacts.

### Src
Users can forgo eda and run src/main.py. This will do the EDA if eda doesn't exist, providing users with data and model artifacts. If these do exist, main.py will use the model - as specified by src/configs/config. The final output of main.py is the creation of a results.csv which reports the capuchin call density per file in data/forest recording. This is the output as specified by the contest. 

## Directory structure
```.
├── artifacts
│   ├── 15_epochs_mels_spectrogram_classification_report.txt
│   ├── 15_epochs_mels_spectrogram_confusion_matrix.png
│   ├── 15_epochs_mels_spectrogram_model_performance_subplots.png
│   ├── 5_epoch_spectrogram_classification_report.txt
│   ├── 5_epoch_spectrogram_confusion_matrix.png
│   ├── 5_epoch_spectrogram_model_performance_subplots.png
│   ├── example_wavelengths.png
│   ├── mels_spectrogram_results.csv
│   ├── models
│   │   ├── mels_spectrogram_model.h5
│   │   └── spectrogram_model.h5
│   ├── positive_negative_spectrogram_examples.png
│   ├── results.csv
│   └── training_wave_lengths.png
├── data
│   ├── Forest Recordings
│   │   ├── ...
│   ├── Parsed_Capuchinbird_Clips
│   │   ├── ...
│   ├── Parsed_Not_Capuchinbird_Clips
│   │   ├── ...
│   └── data_README.md
├── eda
│   ├── __init__.py
│   ├── code_and_data_exploratory_analysis.ipynb
│   ├── eda.py
│   ├── model_dev.py
│   └── utils
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-39.pyc
│       │   └── utils.cpython-39.pyc
│       └── utils.py
└── src
    ├── configs
    │   └── config.json
    ├── eda.py
    ├── main.py
    ├── model_dev_from_config.py
    └── utils
        ├── __init__.py
        └── utils.py
```
12 directories, 938 files

## References
These references were incredibly useful in designing various functions, fundamental understanding, and writing copy as presented in this notebook and in this work. Its important to note that this work is a heavily modified version of the video tutorial found in 1.

1. [Build a Deep Audio Classifier with Python and Tensorflow](https://www.youtube.com/watch?v=ZLIPkmmDJAc&t=2s)
2. ChapGPT, personal communication, 3/4/23 for copy
3. [Understanding the Mel Spectrogram](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53)
4. [How to Create & Understand Mel-Spectrograms](https://importchris.medium.com/how-to-create-understand-mel-spectrograms-ff7634991056)
5. [How to Easily Process Audio on Your GPU with TensorFlow](https://towardsdatascience.com/how-to-easily-process-audio-on-your-gpu-with-tensorflow-2d9d91360f06)

