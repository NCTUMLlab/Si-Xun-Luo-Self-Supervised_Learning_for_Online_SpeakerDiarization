
# Introduction
This is the source code for the Master thesis Self-Supervised Learning for Online SpeakerDiarization from National Chiao Tung University, Taiwan. 

# Self-Supervised Learning for Online SpeakerDiarization

  In this project, we introduced a self-supervised leaning method to train a mapping function to get the predicted initial label of speaker cluster and the speaker feature extraction. As for the audio feature extraction and the change point detection we used online MFCC and online VAD method respectively. 
  
  Self-supervised learning is a specific method of unsupervised learning, it was first used in image representation learning. The self-supervised method was introduced for image clustering, But we further apply the self-supervised learning to predict the initial speaker cluster label and extract the speaker feature in the online speaker diarization task, and then we utilize a suitable online clustering to calculate the final clustering result based on the result of the self-supervised learning and the speaker feature extraction to solve the problem of non-fixed cluster number in this online task. As for the speaker feature extraction we mentioned, there is a speaker level feature extraction based on self-supervised module with TDNN and Bi-LSTM. In our experiment, we used Voxcelb train-data to train the self-supervised learning based speaker feature extraction module and Voxceleb test-data to train the proposed online clustering method. The proposed method is an advanced method based on the the basic model we implemented. 
  
  The basic model is in principle an online version well-trained X-vector based offline speaker diarization model. The advanced online speaker diarization model we proposed made an improvement on these problems and achieved a better performance than the basic model.

# Get starting
## Environment
The developed environment is listed in below

OS : Ubuntu 16.04

CUDA : 11.1

Nvidia Driver : 455.23

Python 3.6.9

Pytorch 1.2.0

## Preprocess

  We use voxceleb1 and voxceleb2 dataset. You should download them and unzip. Or you can use other dataset and follow the same format as voxceleb.

After downloaded, please create the folder named wav and put .wav files into it. Then you should call 
```bash
$ python preprocessing.py 
```
After that, you will get the 'wav_X.npy' and 'wav_Y.npy', which is the input of our model for training.

## Training

Training function and model detail are both in train.py. Please run 
```bash
$ python training.py
```

You can get the 'model_epoch.pkl' during training. The .pkl file is the model parameter for each epoch.



