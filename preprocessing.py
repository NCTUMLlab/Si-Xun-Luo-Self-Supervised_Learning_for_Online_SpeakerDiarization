import os
import numpy as np
from scipy.io import wavfile
import librosa

path_now = os.getcwd()


def readWav(path,s_r=8000):

    sig, sr = librosa.load(path,sr=s_r)
    sig = sig[::2]

    i = 1
    output = [] 

    while i*s_r < len(sig) :
        output.append( sig[(i-1)*s_r:i*s_r] ) 
        i+=1

    return output

def mfcc(x,s_r=8000):

    return librosa.feature.mfcc(y=x, sr=8000,n_mfcc=23,n_fft=200, hop_length=80).transpose((1,0))

def createData ( path ) :

    X = []

    Y = []

    i = 0    

    folders = os.listdir (path)

    for folder in sorted(folders) :

        path_ = path + folder + '/'

        folders_ = os.listdir (path_)

        for folder_ in sorted (folders_)  :  

            path__ = path_ + folder_ + '/'

            wavs = os.listdir (path__)

            for wav in sorted(wavs) :

                x = [mfcc(i) for i in readWav( path__+wav )]
                
                X = X + x
                
                Y = Y + [i for t in range( len(x) )]

        i+=1    
        
    return np.array(X),np.array(Y)
        

def main():

    X,Y = createData (path_now+'/wav/')

    print (X.shape)

    print (Y.shape)

    np.save('wav_X',X)

    np.save('wav_Y',Y)


if __name__ == '__main__':

    main()

