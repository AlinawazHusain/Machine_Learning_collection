import torch 
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self , z_dim , out_ch , features ):
        super(Generator , self ).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(z_dim , features*8 , 4 , 2 , 0),
            nn.BatchNorm2d(features*8),
            nn.LeakyReLU(0.2),
            self.ConvBlock(features*8, features*4),
            self.ConvBlock(features*4, features*2),
            self.ConvBlock(features*2, features),
            nn.ConvTranspose2d(features, out_ch , 4,2,1),
            nn.Tanh()
        )

    def ConvBlock(self , in_ch , out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch , out_ch , 4 , 2 , 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(0.2)
        )
    
    def forward(self , x):
        return self.layers(x)





class Descriminator(nn.Module):
    def __init__(self , in_ch , features):
        super(Descriminator , self).__init__()
        self.layers = nn.Sequential(
            self.convBlock(in_ch , features*8),
            self.convBlock(features*8 , features*4),
            self.convBlock(features*4 , features*2),
            self.convBlock(features*2 , features),
            nn.Conv2d(features , 1 , 4 , 2 , 0),
            nn.Sigmoid()

        )

    def convBlock(self , in_ch , out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch , out_ch , 4 , 2 , 1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self , x):
        return self.layers(x)
    



def initialize_weights (model):
    for m in model.modules():
        if isinstance(m , (nn.Conv2d , nn.ConvTranspose2d , nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data , 0.0 , 0.02)


if __name__ == '__main__':
    model = Generator(100 , 1 , 16)
    des = Descriminator(1 , 16)
    x = torch.randn(16 , 100 , 1 , 1)
    y = model(x)
    print(y.shape)
    print(des(y).reshape(16 ,-1).shape)

