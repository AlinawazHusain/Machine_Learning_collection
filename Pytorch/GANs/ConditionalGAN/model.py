import torch 
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self , z_dim , out_ch , features ,num_classes ,img_size , embed_dim ):
        super(Generator , self ).__init__()
        self.img_size = img_size
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(z_dim+embed_dim, features*8 , 4 , 2 , 0),
            nn.BatchNorm2d(features*8),
            nn.LeakyReLU(0.2),
            self.ConvBlock(features*8, features*4),
            self.ConvBlock(features*4, features*2),
            self.ConvBlock(features*2, features),
            nn.ConvTranspose2d(features, out_ch , 4,2,1),
            nn.Tanh()
        )

        self.embedding = nn.Embedding(num_classes , embed_dim)

    def ConvBlock(self , in_ch , out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch , out_ch , 4 , 2 , 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(0.2)
        )
    
    def forward(self , x , label):
        embedded = self.embedding(label).unsqueeze(2).unsqueeze(3)
        input_noise = torch.cat([x , embedded] , dim = 1)
        return self.layers(input_noise)





class Critic(nn.Module):
    def __init__(self , in_ch , features , num_classes , img_size ):
        super(Critic , self).__init__()
        self.img_size = img_size
        self.layers = nn.Sequential(
            self.convBlock(in_ch+1 , features*8),
            self.convBlock(features*8 , features*4),
            self.convBlock(features*4 , features*2),
            self.convBlock(features*2 , features),
            nn.Conv2d(features , 1 , 4 , 2 , 0),
            nn.Sigmoid()

        )
        self.embedding = nn.Embedding(num_classes , img_size*img_size)

    def convBlock(self , in_ch , out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch , out_ch , 4 , 2 , 1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self , x , label):
        embedded = self.embedding(label).reshape(label.shape[0] , 1 , self.img_size , self.img_size)
        input_img = torch.cat([x , embedded] , dim = 1)
        return self.layers(input_img)
    



def initialize_weights (model):
    for m in model.modules():
        if isinstance(m , (nn.Conv2d , nn.ConvTranspose2d , nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data , 0.0 , 0.02)





    