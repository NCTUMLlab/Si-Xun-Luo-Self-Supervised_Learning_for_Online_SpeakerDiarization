import torch
import torch.nn as nn
import torch.nn.functional as F

class TDNN(nn.Module):
    
    def __init__(
                    self, 
                    input_dim=23, 
                    output_dim=512,
                    context_size=5,
                    stride=1,
                    dilation=1,
                    batch_norm=False,
                    dropout_p=0
                ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Affine transformation not applied globally to all frames but smaller windows with local context
        batch_norm: True to include batch normalisation after the non linearity
        
        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
      
        self.kernel = nn.Linear(self.input_dim*self.context_size, self.output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)
        
    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''

        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)


        # Unfold input into smaller temporal contexts
        x = F.unfold(
                        x, 
                        (self.context_size, self.input_dim), 
                        stride=(1,self.input_dim), 
                        dilation=(self.dilation,1)
                    )

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1,2)
        if self.dropout_p:
            x = self.drop(x)
        x = self.kernel(x)
        x = self.nonlinearity(x)
        
        if self.batch_norm:
            x = x.transpose(1,2)
            x = self.bn(x)
            x = x.transpose(1,2)

        return x

class StatsPooling(nn.Module):
    def __init__(self):
        super(StatsPooling,self).__init__()

    def forward(self,varient_length_tensor):
        mean = varient_length_tensor.mean(dim=1)
        std = varient_length_tensor.std(dim=1)
        return mean+std


class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.hidden1 = nn.Linear(512,512)
        self.hidden2 = nn.Linear(512,512)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.hidden1(x)#F.relu( self.hidden1(x))
        x = self.dropout(x)
        x = self.hidden2(x)
        return x

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.frame1 = TDNN(input_dim=23, output_dim=512, context_size=5, dilation=1)
        self.frame2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2)
        self.frame3 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=3)
        self.frame4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1)
        self.frame5 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1)
        self.pooling = StatsPooling()
        self.fully = FullyConnected()
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x1 = self.frame1(x)
        x2 = self.frame2(x1)
        x3 = self.frame3(x2)
        x4 = self.frame4(x3)
        x5 = self.frame5(x4)
        x6 = self.pooling(x5)
        x7 = self.fully(x6) 
        x7 = self.softmax(x7)
        return x7

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



