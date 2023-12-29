import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
import  torchvision.datasets as datasets
import torchvision.transforms as transform
import model
from torch.utils import tensorboard


device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 2e-4
b_size = 64
epochs = 5
features_g = 64
features_d = 64
img_channels = 1 # MNist B&w imges
img_size = 64
z_dim = 100

Transform = transform.Compose(
    [
        transform.Resize((img_size , img_size)),
        transform.ToTensor(),
        transform.Normalize([0.5]*img_channels , [0.5]*img_channels)
    ]
)


gen = model.Generator(z_dim , img_channels , features_g).to(device)

des = model.Descriminator(img_channels , features_d).to(device)
model.initialize_weights(gen)
model.initialize_weights(des)


optim_gen = optim.Adam(gen.parameters() , lr = learning_rate , betas = (0.5 , 0.999))
optim_des = optim.Adam(des.parameters() , lr = learning_rate , betas = (0.5 , 0.999))

loss_fn = nn.BCELoss().to(device)


data = datasets.MNIST(root = 'data/' , train = True , download = True , transform = Transform)
loader = DataLoader(data , b_size , shuffle = True)

fake_writer = SummaryWriter(f'logs/fake')
real_writer = SummaryWriter(f'logs/real')
step = 0

gen.train()
des.train()
for epoch in range(epochs):
    p_bar = tqdm(loader , desc = f"Epoch {epoch}/{epochs}")

    for batch_idx , (real , label) in enumerate(p_bar):
        real = real.to(device)
        label = label.to(device)

        current_b_size = real.shape[0]
        noise = torch.randn(current_b_size ,z_dim,1,1).to(device)


        des.zero_grad()

        fake = gen(noise)

        fake_out = des(fake.detach()).reshape(-1)
        real_out = des(real).reshape(-1)

        real_labels = torch.full((real_out.size(0),), 1.0, device=device)
        fake_labels = torch.full((fake_out.size(0),), 0.0, device=device)

        des_loss_fake = loss_fn(fake_out , fake_labels)
        des_loss_real = loss_fn(real_out , real_labels)

        des_loss_fake.backward()
        des_loss_real.backward()

        optim_des.step()


        gen.zero_grad()
        output = des(gen(noise)).reshape(-1)
        gen_loss = loss_fn(output , real_labels)
        gen_loss.backward()
        optim_gen.step()

        p_bar.set_postfix({'GenLoss' :gen_loss.item() , 'DesLossReal': des_loss_real.item() , 'DesLossFake': des_loss_fake.item()})


        if batch_idx % 100 == 0:
            with torch.no_grad():
                gen.eval()
                des.eval()
                test_noise = torch.randn(current_b_size , z_dim , 1 ,1)
                fake_img = gen(test_noise)


                img_grid_fake = torchvision.utils.make_grid(fake[:20], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:20], normalize=True)

                fake_writer.add_image('Fake', img_grid_fake , global_step = step)
                real_writer.add_image('Real' , img_grid_real , global_step = step)

                step = step+1

            gen.train()
            des.train()


torch.save({
    'generator' : gen,
    'descriminator' : des,
    'gen_state_dict' : gen.state_dict(),
    'des_state_dict' : des.state_dict(),
    'gen_optimizer' :optim_gen.state_dict(),
    'des_optimizer' : optim_des.state_dict(),
} , "DCGAN.pth.tar"
)
