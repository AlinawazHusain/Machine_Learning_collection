import model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transform
from tqdm import tqdm

input_size = 28
seq_len = 28 
num_layers = 2
hidden_size = 128
num_classes = 10 #MNIST
EPOCH = 1


Model = model.LSTM(input_size , hidden_size , num_layers , num_classes)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(Model.parameters() , lr = 0.001)

train_dataset = MNIST(root = 'dataset/' , transform= transform.ToTensor() , train = True , download = True)
test_dataset = MNIST(root = 'dataset/' , transform= transform.ToTensor() , train = False , download = True)

train_loader = DataLoader(train_dataset , batch_size = 8 , shuffle = True )
test_loader = DataLoader(test_dataset , batch_size = 8 , shuffle = True )


Model.train()
for epoch in range(EPOCH):
    for batch_idx , (x , y) in enumerate(tqdm(train_loader , desc = f"epoch {epoch+1}/1")):
        x = x.squeeze(1)
        pred = Model(x)
        loss = loss_fn(pred , y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


with torch.no_grad():
    Model.eval()
    correct = 0
    total = 0
    for batch_idx , (x , y) in enumerate(tqdm(test_loader , desc = f"epoch {epoch+1}/6")):
        x = x.squeeze(1)
        pred = Model(x)
        total = total+8

        correct += (pred.argmax(1) == y).sum()

    print(f"accuracy = {(correct / total)*100}")

