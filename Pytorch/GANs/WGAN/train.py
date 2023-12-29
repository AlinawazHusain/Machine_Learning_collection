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
learning_rate = 5e-5
b_size = 64
epochs = 5
features_g = 64
features_d = 64
img_channels = 1 # MNist B&w imges
img_size = 64
z_dim = 100
critic_iterations = 5
weight_clip = 0.01

Transform = transform.Compose(
    [
        transform.Resize((img_size , img_size)),
        transform.ToTensor(),
        transform.Normalize([0.5]*img_channels , [0.5]*img_channels)
    ]
)


gen = model.Generator(z_dim , img_channels , features_g).to(device)

critic = model.Critic(img_channels , features_d).to(device)
model.initialize_weights(gen)
model.initialize_weights(critic)


optim_gen = optim.RMSprop(gen.parameters() , lr = learning_rate)
optim_critic = optim.RMSprop(critic.parameters() , lr = learning_rate)

loss_fn = nn.BCELoss().to(device)


data = datasets.MNIST(root = 'data/' , train = True , download = True , transform = Transform)
loader = DataLoader(data , b_size , shuffle = True)

fake_writer = SummaryWriter(f'logs/fake')
real_writer = SummaryWriter(f'logs/real')
step = 0

gen.train()
critic.train()
for epoch in range(epochs):
    p_bar = tqdm(loader , desc = f"Epoch {epoch}/{epochs}")

    for batch_idx , (real , label) in enumerate(p_bar):
        real = real.to(device)
        label = label.to(device)


        for i in range(critic_iterations):
            current_b_size = real.shape[0]
            noise = torch.randn(current_b_size ,z_dim,1,1).to(device)


            critic.zero_grad()

            fake = gen(noise)

            fake_out =critic(fake).reshape(-1)
            real_out = critic(real).reshape(-1)

            critic_loss = -(torch.mean(real_out) - torch.mean(fake_out))

            critic_loss.backward(retain_graph=True)
            optim_critic.step()

            for p in critic.parameters():
                p.data.clamp_(-weight_clip , weight_clip)


        gen.zero_grad()
        output = critic(gen(noise)).reshape(-1)
        gen_loss = -torch.mean(output)
        gen_loss.backward()
        optim_gen.step()

        p_bar.set_postfix({'GenLoss' :gen_loss.item() , 'Critic_loss': critic_loss.item()})


        if batch_idx % 100 == 0:
            with torch.no_grad():
                gen.eval()
                critic.eval()
                test_noise = torch.randn(current_b_size , z_dim , 1 ,1)
                fake_img = gen(test_noise)


                img_grid_fake = torchvision.utils.make_grid(fake[:20], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:20], normalize=True)

                fake_writer.add_image('Fake', img_grid_fake , global_step = step)
                real_writer.add_image('Real' , img_grid_real , global_step = step)

                step = step+1

            gen.train()
            critic.train()


torch.save({
    'generator' : gen,
    'critic' : critic,
    'gen_state_dict' : gen.state_dict(),
    'critic_state_dict' : critic.state_dict(),
    'gen_optimizer' :optim_gen.state_dict(),
    'critic_optimizer' : optim_critic.state_dict(),
} , "WGAN.pth.tar"
)
