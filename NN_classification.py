#python 
import torch.nn as nn
import torch

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

import time

#Transform data
train_transforms = transforms.Compose([
    transforms.Resize(size=(180 , 180) ) ,
    transforms.ToTensor () ,])

#Load dataset
train_dataset = ImageFolder(root="flower_photos", transform=train_transforms )
train_loader = DataLoader(train_dataset, batch_size=128)

#Input=(batch_size, 3, 180, 180)
#Conv output = (w-k+2P)/s+1

#Define network using convolutions, max pooling, fully connected, ReLu and softmax layers
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 16 output channels, 3x3 square convolution kernel
        #shape=(batch_size, 3, 180, 180)
        self.conv1=nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=1,padding=1)
        #shape=(batch_size, 8, 180, 180)
        self.pool=nn.MaxPool2d(kernel_size=2)
        #shape=(batch_size, 8, 90, 90)
        self.conv2=nn.Conv2d(in_channels=8,out_channels=20,kernel_size=3,stride=1,padding=1)
        #shape=(batch_size, 20, 90, 90)
       
        self.fc1=nn.Linear(in_features=20*90*90,out_features=120)
        self.relu1=nn.ReLU()
        self.fc2=nn.Linear(in_features=120,out_features=80)
        self.relu2=nn.ReLU()
        self.fc3=nn.Linear(in_features=80,out_features=5)

        self.sm=nn.Softmax(dim=1)

    def forward(self,input):
        output=self.conv1(input)
        output=self.pool(output)
        output=self.conv2(output)

        output=output.view(-1,20*90*90) #Flatten

        output=self.fc1(output)
        output=self.relu1(output)
        output=self.fc2(output)
        output=self.relu2(output)
        output=self.fc3(output)

        output=self.sm(output)
        return output

#Create model and get summary
model = Net()
print(model)

#Define loss function and optimizer
loss_fn= nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

losses = []
t0 = time.time()

print(y)

#Train model
for epoch in range(15):
    for (x,y) in train_loader:
        y=torch.nn.functional.one_hot(y,num_classes=5).float()
        opt.zero_grad()
        y_hat=model(x)
        loss=loss_fn(y_hat,y)
        loss.backward()
        opt.step()
    losses.append(loss.detach())
    print(f'Epoch: {epoch}, loss:{loss.detach()}')

t1 = time.time()
print(f'Total time: {t1-t0}')

#Show results
plt.plot(losses)
plt.savefig('Flower cross-ent loss.png')
plt.show()
