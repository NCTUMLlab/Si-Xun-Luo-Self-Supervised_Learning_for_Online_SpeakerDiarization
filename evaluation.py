import webrtcvad
import numpy as np
import random
import torch
import torch.nn as nn
import time
import librosa
from tqdm import tqdm
import os
from scipy.io import wavfile
import pydub
from Layer import *

vad = webrtcvad.Vad(3)
sample_rate = 8000
node100ms = int(sample_rate/10)

import webrtcvad
vad = webrtcvad.Vad(3)

sample_rate = 8000

def detect(frame_now):
    
    times = 0
    for i in range(5):
        frame = frame_now[i*160:160*(i+1)] 
        if vad.is_speech(frame,8000):
            times+=1
    if times>2:
        return True
    return False

model = torch.load('example.pkl').cuda(0)

print ()

def evaluation(file):
    
    straight = 3
    
    test = readWav2(file)
    
    name = file.split('/')[-1].split('.')[0]
    
    node100ms = 800
    
    i=1
    
    predict_frame = np.zeros(8000)

    result = [torch.zeros(1300)]
    
    detect_ = [True]

    while(node100ms*i<len(test)): 
        
        

        result.append(torch.zeros(1300)) 
        frame_now = test[node100ms*(i-1):node100ms*i]

        #detect_.append( detect(frame_now) )

        predict_frame = np.concatenate((predict_frame[800:8000], frame_now), axis=None) 
        probability_distribution = model.predict(predict_frame) 
        
        detect_.append( detect(frame_now) )
        
        k = 0 
        while(i-k>=0 and k<10): 
            result[i-k] = result[i-k] + probability_distribution 
            k+=1 
        i+=1
    who = np.zeros(len(result),dtype=int)
    output = np.zeros(len(result),dtype=int)
    dic = {}
    for i in range( 1,len(who) ):
        who[i] = int(torch.argmax(result[i]))
        k = 0 
        while(i-k>=0 and k<straight):
            if who[i] != who [i-k]:
                break
            k+=1
        if k==straight:
            output[i] = who[i]
        else:
            output[i] = output[i-1]
    f = open('x.rttm', 'a')
    for i in range(1,len(output)):
        if not detect_[i]:
            continue
        string = 'SPEAKER '+ name+' 0 '+ str((i-1)*0.1)+' '+ str(0.1) + ' <NA> <NA> '+ str(output[i]) + ' <NA> <NA>\n'
        f.write(string)

def readWav2(path,s_r=8000):
    
    sig, sr = librosa.load(path,sr=s_r)
    
    return sig

for file in sorted(list(os.listdir('evaluation_wav'))):
    
    print (file)
    evaluation('./evaluation_wav/'+file)
