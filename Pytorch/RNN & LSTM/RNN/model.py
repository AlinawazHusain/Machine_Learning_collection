import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self , input_size , hidden_size ,num_layers , seq_length , num_classes  ):
        super(RNN , self ).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size ,hidden_size , num_layers  , batch_first= True)
        ## RNN can be replace by GRU for better accuracy ##
        self.linear = nn.Linear(hidden_size*seq_length , num_classes)

    def forward(self , x):
        h0 = torch.zeros(self.num_layers , x.size(0) ,self.hidden_size )
        out , _ = self.rnn(x , h0)
        out = out.reshape(out.shape[0] , -1)
        return self.linear(out)

    
