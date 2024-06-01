import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self , input_size , hidden_size ,num_layers , num_classes  ):
        super(LSTM , self ).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size ,hidden_size , num_layers  , batch_first= True)
        self.linear = nn.Linear(hidden_size , num_classes)

    def forward(self , x):
        h0 = torch.zeros(self.num_layers , x.size(0) ,self.hidden_size )
        c0 = torch.zeros(self.num_layers , x.size(0) ,self.hidden_size )
        out , _ = self.rnn(x , (h0 , c0))
        out = out[: , -1 , :]
        # out = out.reshape(out.shape[0] , -1)
        return self.linear(out)