import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self , in_size , out_size):
        super(Model , self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_size , in_size*2),
            nn.ReLU(),
            nn.Linear(in_size*2 , out_size *8),
            nn.ReLU(),
            nn.Linear(out_size *8 , out_size *4),
            nn.ReLU(),
            nn.Linear(out_size *4 , out_size),
            nn.Softmax(),
        )


    def forward(self , x):
        return self.layers(x)
    



if __name__ == '__main__':
    model = Model(728 , 10)
    x = torch.randn(12 , 728)
    print(model(x).shape)