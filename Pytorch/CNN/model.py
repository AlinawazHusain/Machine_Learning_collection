import torch
import torch.nn as nn


#================================================================================#
#                         Layers for 64x64 image size                            #
#================================================================================# 


class CNN(nn.Module):
    def __init__(self , in_ch , num_classes ,features = 16):
        super(CNN , self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_ch , features*8 , 3 , 2 , 1),
            nn.ReLU(),
            nn.Conv2d(features*8 , features*4 , 3 , 2 , 1),  
            nn.ReLU(),
            nn.Conv2d(features*4 , features*2 , 3 , 2 ,1),  
            nn.ReLU() ,
            nn.Conv2d(features*2 , features , 3 , 2 ,1),  
            nn.ReLU(),
            nn.Conv2d(features , features//2 , 3 , 2 ,1),  
            nn.ReLU(),
            nn.Conv2d(features//2 , num_classes , 3 , 2 ,1),  
            nn.Softmax()
        )


    def forward(self , x):
        return self.layers(x)
    








if __name__ == '__main__':
    x = torch.randn(3 , 64 , 64)
    model = CNN(3 , 1 , 16)
    print(model(x).shape)