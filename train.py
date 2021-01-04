from pynvml import *
import numpy as np
import torch
import torch.nn as nn
from loadData import *
from createWavXandY import *
from TDNN_ import *
import time
from tqdm import tqdm
import librosa
batch = 512
epoch = 50 


nvmlInit()
print ("Driver Version:", nvmlSystemGetDriverVersion())
deviceCount = nvmlDeviceGetCount()
for i in range(deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    print ("Device", i, ":", nvmlDeviceGetName(handle))

class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()
        
        #self.mfcc = MFCC(sample_rate=8000, n_mfcc= 23)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.TDNN = MyModel()
        
        self.softmax = nn.Softmax(dim=1)
    
    def load_(self):
        
        self.TDNN = torch.load('best.pkl')

    def save_(self,epoch_):
        
        torch.save(self.TDNN,str(epoch_)+'_model.pkl')
        
        
    def forward(self, x ):

        one = torch.squeeze(x[:,0:1,:,:])

        other = torch.squeeze(x[:,1:2,:,:])

        one  = self.TDNN(one)

        other= self.TDNN(other)

        output = self.cos(one,other)

        return output 

def main():

    model = Model()
    model=model.cuda(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    cal_loss = torch.nn.MSELoss()

    for epoch_ in range(epoch):
    
        if epoch_%5 == 0:
            trainX , trainY = maketrainXandY(int(epoch_/5))
            testX , testY = maketestXandY()
            trainX = torch.from_numpy(trainX).type(torch.FloatTensor)
            trainY = torch.from_numpy(trainY).type(torch.FloatTensor)
            torch_train_dataset = torch.utils.data.TensorDataset(trainX,trainY)
            train_loader = torch.utils.data.DataLoader(dataset=torch_train_dataset, batch_size=batch, shuffle=True, num_workers=2, drop_last=True)

            testX = torch.from_numpy(testX).type(torch.FloatTensor)
            testY = torch.from_numpy(testY).type(torch.FloatTensor)
            torch_test_dataset = torch.utils.data.TensorDataset(testX,testY)
            test_loader = torch.utils.data.DataLoader(dataset=torch_test_dataset, batch_size=batch, shuffle=True, num_workers=2, drop_last=True)
            
            
        
        
        print ('epoch : ', epoch_+1 )
        train_loss = 0
        start = time.time()
        for step, (b_x, b_y) in tqdm(enumerate(train_loader)):

            b_x = b_x.cuda(1)
            b_y = b_y.cuda(1)
            output = model(b_x)
            loss = cal_loss(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            if step == len (train_loader)-1:
                print( 'train_loss = ' ,  train_loss/len (train_loader) )
        end = time.time()
        print ( 'training time : ',end - start , ' s ')
            
        with torch.no_grad() :
            acc_num=0
            test_loss = 0
            start = time.time()
            for step, (b_x, b_y) in enumerate(test_loader):

                b_x = b_x.cuda(1)
                b_y = b_y.cuda(1)
                output = model(b_x)
                loss = cal_loss(output,b_y)
                output = output.cpu().numpy()
                b_y = b_y.cpu().numpy()
                for i in range(len(output)) :
                    if ( output[i]>0.5 and b_y[i]==1 ) or output[i]<0.5 and b_y[i]==0:
                        acc_num+=1
                    
                test_loss+=loss.item()
                if step == len (test_loader)-1:
                    print( 'test_loss = ' ,  test_loss/len (test_loader) )
            print ('acc = ' , acc_num/(batch*len (test_loader)) )
            end = time.time()
            print ( 'testing time : ',end - start , ' s ')
            model.save_(epoch_+1)

if __name__ == '__main__':

    main()


