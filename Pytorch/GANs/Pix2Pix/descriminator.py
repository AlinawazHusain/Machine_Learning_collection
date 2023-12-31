import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self , in_ch , out_ch , st = 2):
        super(ConvBlock ,self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_ch , out_ch , 4 , st , bias = False , padding_mode='reflect'),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )
    
    
    def forward(self , x):
        return self.layers(x)
    


class Descriminator(nn.Module):
    def __init__(self , in_ch = 3 , features = [64 , 128 , 256 , 512] ):
        super(Descriminator , self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_ch*2 , features[0] , 4 , 2 , 1,padding_mode = 'reflect' , bias = False),
            nn.LeakyReLU(0.2)
        )

        channels = features[0]
        layers = []
        for feature in features[1:]:
            layers.append(ConvBlock(channels , feature , st = 1 if feature == features[-1] else 2))

            channels = feature

        layers.append(nn.Conv2d(features[-1] , 1 , 4 , 1 , 1 , padding_mode='reflect'))
        self.model = nn.Sequential(*layers)


    def forward(self , x , y):
        input = torch.cat((x , y) ,dim = 1)
        input = self.initial(input)
        return self.model(input)




if __name__ == '__main__':

    x = torch.randn(2 , 3, 256 , 256)
    y = torch.randn(2 , 3, 256 , 256)
    model = Descriminator()
    print(model(x , y).shape)

