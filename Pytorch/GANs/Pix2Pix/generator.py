import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self , in_ch , out_ch , activation = 'ReLU' , down = True , use_dropout = False):
        super(ConvBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_ch , out_ch , 4 , 2 , 1 , padding_mode='reflect' , bias = False)
            if down
            else nn.ConvTranspose2d(in_ch , out_ch , 4 , 2 ,1 ,bias = False),

            nn.BatchNorm2d(out_ch),
            nn.ReLU() if activation == 'ReLU' else nn.LeakyReLU(0.2)
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout2d(0.5)

    def forward(self , x):
        x = self.layer(x)
        return self.dropout(x) if self.use_dropout else x
    



class Generator(nn.Module):
    def __init__(self , in_ch = 3 , features = 64):
        super(Generator ,self).__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_ch , features ,4 , 2 , 1 , padding_mode='reflect' ),
            nn.LeakyReLU(0.2)
        )

        self.down1 = ConvBlock(features , features*2 , 'Leaky' , down = True , use_dropout= False)
        self.down2 = ConvBlock(features*2 , features*4 , 'Leaky' , down = True , use_dropout= False)
        self.down3 = ConvBlock(features*4 , features*8 , 'Leaky' , down = True , use_dropout= False)
        self.down4 = ConvBlock(features*8 , features*8 , 'Leaky' , down = True , use_dropout= False)
        self.down5 = ConvBlock(features*8 , features*8 , 'Leaky' , down = True , use_dropout= False)
        self.down6 = ConvBlock(features*8 , features*8 , 'Leaky' , down = True , use_dropout= False)
        
        self.bottleNeck = nn.Sequential(
            nn.Conv2d(features*8 , features*8 , 4 , 2, 1 , padding_mode='reflect'),
            nn.ReLU()
        )

        self.up1 = ConvBlock(features*8 , features*8 , 'ReLU' , down = False , use_dropout= True)
        self.up2 = ConvBlock(features*8*2 , features*8 , 'ReLU' , down = False , use_dropout= True)
        self.up3 = ConvBlock(features*8*2 , features*8 , 'ReLU' , down = False , use_dropout= True)
        self.up4 = ConvBlock(features*8*2 , features*8 , 'ReLU' , down = False , use_dropout= True)
        self.up5 = ConvBlock(features*8*2 , features*4 , 'ReLU' , down = False , use_dropout= True)
        self.up6 = ConvBlock(features*4*2 , features*2 , 'ReLU' , down = False , use_dropout= True)
        self.up7 = ConvBlock(features*2*2 , features , 'ReLU' , down = False , use_dropout= True)


        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2 , in_ch , 4 , 2, 1 ),
            nn.Tanh()
        )



    def forward(self , x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        
        bottleneck = self.bottleNeck(d7)

        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat((up1 , d7) , dim = 1))
        up3 = self.up3(torch.cat((up2 , d6) , dim = 1))
        up4 = self.up4(torch.cat((up3 , d5) , dim = 1))
        up5 = self.up5(torch.cat((up4 , d4) , dim = 1))
        up6 = self.up6(torch.cat((up5 , d3) , dim = 1))
        up7 = self.up7(torch.cat((up6 , d2), dim = 1))
        

        return self.final_up(torch.cat((up7 , d1) , dim = 1))
    



if __name__ == '__main__':
    x = torch.randn(2 , 3 , 256 , 256)

    model = Generator()
    print(model(x).shape)
        