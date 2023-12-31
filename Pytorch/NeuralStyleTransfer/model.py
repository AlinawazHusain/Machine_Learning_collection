import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transform
import torchvision.models as models
from PIL import Image
from torchvision.utils import save_image
from tqdm import trange



device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_size = 256

class Model(nn.Module):
    def __init__(self ):
        super(Model , self ).__init__()

        self.vgg = models.vgg19(pretrained = True).features[:29]
        self.needed_layers = [0 , 5 , 10 , 19 , 28]
    

    def forward(self , x):
        features = []

        for layer_idx , layer in enumerate(self.vgg):
            x = layer(x)

            if layer_idx in self.needed_layers:
                features.append(x)
        
        return features



def load_image(image):
    img = Image.open(image)
    img = Transform(img).unsqueeze(0)
    return img.to(device)


Transform = transform.Compose(
    [
        transform.Resize((image_size , image_size)),
        transform.ToTensor()
    ]
)


img = load_image('Model/data/annahathaway.png')
style = load_image('Model/data/style.jpg')

generated = img.clone().requires_grad_(True)


model = Model().to(device).eval()

total_steps = 6001
learning_rate = 0.001
alpha = 1
beta = 0.01
optimizer = optim.Adam([generated], lr=learning_rate)


for step in trange(total_steps):
    
    generated_features = model(generated)
    original_img_features = model(img)
    style_features = model(style)

    
    style_loss = original_loss = 0

   
    for gen_feature, orig_feature, style_feature in zip(
        generated_features, original_img_features, style_features
    ):

       
        batch_size, channel, height, width = gen_feature.shape
        original_loss += torch.mean((gen_feature - orig_feature) ** 2)
        
        G = gen_feature.view(channel, height * width).mm(
            gen_feature.view(channel, height * width).t()
        )
        
        A = style_feature.view(channel, height * width).mm(
            style_feature.view(channel, height * width).t()
        )
        style_loss += torch.mean((G - A) ** 2)

    total_loss = alpha * original_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        print(total_loss)
        save_image(generated, "generated.png")