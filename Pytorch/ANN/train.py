import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transform 
from torch.utils.data import DataLoader
import model
from tqdm import tqdm



device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-3
b_size = 64
epochs = 10
channels = 1 
input_size = 28*28
classes = 10

Transform = transform.Compose(
    [
        transform.Resize((28 , 28)),
        transform.ToTensor(),
        transform.Normalize([0.5]*channels , [0.5]*channels)
    ]
)


model = model.Model(input_size , classes).to(device)
optimizer = optim.Adam(model.parameters() , lr = learning_rate)
loss_fn = nn.CrossEntropyLoss().to(device)

train_data = datasets.MNIST(root = 'data/' , train = True , transform=Transform ,download = True)
test_data = datasets.MNIST(root = 'data/' , train = False , transform=Transform ,download = True)

train_loader = DataLoader(train_data , b_size , shuffle = True)
test_loader = DataLoader(test_data , b_size , shuffle = True)


model.train()

for epoch in range(epochs):
    p_bar = tqdm(train_loader)

    for _ , (image , label) in enumerate(p_bar):
        image = image.reshape((image.shape[0] , 28*28)).to(device)
        label = label.to(device)

        y = model(image)

        loss = loss_fn(y,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        p_bar.set_postfix({'Loss ': loss})

    
    if epoch%3 == 0:
        with torch.no_grad():
            model.eval()
            test_pbar = tqdm(test_loader)
            for _ , (img , label) in enumerate(test_pbar):
                img = img.reshape(img.shape[0] , 28*28).to(device)
                label = label.to(device)

                output = model(img)
                output = torch.argmax(output , dim = 1)

                total = output.shape[0]
                accurate = (output == label).sum()
                accuracy = (accurate/total)*100

            print(f"accuracy = {accuracy}")
            model.train()
        

torch.save({
    'model' : model,
    'state_dict':model.state_dict(),
    'optimizer' : optimizer.state_dict()
}, 'Trained_model.pth.tar'
)