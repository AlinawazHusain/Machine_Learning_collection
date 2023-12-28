import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import model
import torchvision.transforms as transform
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-4
b_size = 64
epochs = 10
in_channels = 1  #for MNist for RGB change to 3
num_classes = 10 #for MNist
features = 16
save_model = False


Transform = transform.Compose(
    [
        transform.Resize((64 , 64)),
        transform.ToTensor(),
        transform.Normalize([0.5]*in_channels , [0.5]*in_channels)
    ]
)


model = model.CNN(in_channels , num_classes , features).to(device)
optimizer = optim.Adam(model.parameters() , lr = learning_rate )
loss_fn = nn.CrossEntropyLoss().to(device)


train_data = datasets.MNIST(root = 'data/' , train = True , download = True , transform = Transform)
test_data = datasets.MNIST(root = 'data/' , train = False , transform = Transform, download = True)
train_loader = DataLoader(train_data , batch_size = b_size , shuffle = True)
test_loader = DataLoader(test_data , batch_size = b_size , shuffle = True)


model.train()
for epoch in range(epochs):
    pbar = tqdm(train_loader , desc = f'epoch {epoch}/{epochs}')

    for batch_idx , (image , label ) in enumerate(pbar):
        image = image.to(device)
        label = label.to(device)

        y = model(image)
        y = y.squeeze (-1).squeeze(-1)
        loss = loss_fn(y , label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix({'loss ': loss})


    if epoch%3 == 0:
        print("Testing output and accuracy")
        with torch.no_grad():
            model.eval()
            test_pbar = tqdm(test_loader , desc = f'Testing outputs')
            for _ , (image , label) in enumerate(test_pbar):
                image = image.to(device)
                label = label.to(device)

                outputs = model(image).squeeze (-1).squeeze(-1)
                outputs = torch.argmax(outputs , dim = 1)

                total = outputs.shape[0]
                accurate = (outputs == label).sum()

                accuracy = (accurate/total)*100

            print(f'Accuracy : {accuracy}')
            model.train()



if save_model:
    torch.save({
        'model': model,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'trained_model.pth.tar')


