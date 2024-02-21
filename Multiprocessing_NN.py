#python Ex10.py
import torch.nn as nn
import torch

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch.multiprocessing import Process

import time
from datetime import datetime
import random

random.seed(10)

from torchvision import transforms

def get_mean_std(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_pixels = 0
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.mean(axis=(0, 2, 3)).sum()
        std += images.std(axis=(0, 2, 3)).sum()

    mean /= num_pixels
    std /= num_pixels

    return mean, std




#Define network using convolutions, max pooling, fully connected, ReLu and softmax layers
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=1,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2)
        self.conv2=nn.Conv2d(in_channels=8,out_channels=20,kernel_size=3,stride=1,padding=1)
       
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
    
def train(data_loader, model, loss_fn, optimizer, epochs, workers):
    log_dir = " logs/" + datetime.now().strftime("%m.%d-%H:%M:%S") + 'e' + str(epochs) + 'w' + str(workers)
    log_dir=log_dir.replace('.','-')
    log_dir=log_dir.replace(':','-')

    writer = SummaryWriter(log_dir)
    for epoch in range(epochs):
        for data, labels in data_loader:
            labels=nn.functional.one_hot(labels,num_classes=5).float()
            optimizer.zero_grad()
            loss_fn(model(data), labels).backward()
            optimizer.step()
        writer.add_scalar("epoch_loss", loss_fn(model(data), labels).detach() / len(data_loader), epoch) 
        writer.flush()
        print(f'Epoch: {epoch+1}, loss:{loss_fn(model(data), labels).detach()}')

def test(data, model, loss_fn):
    losses=[]
    for data, labels in data:
        labels=nn.functional.one_hot(labels,num_classes=5).float()
        losses=loss_fn(model(data),labels)
    return losses

if __name__ == '__main__':
    workers=2
    epochs=20
    t0=time.time()

    #Transform data
    train_transforms = transforms.Compose([
        transforms.Resize(size=(180 , 180) ) ,
        transforms.ToTensor () ,])

    #Load dataset
    dataset = ImageFolder(root="flower_photos", transform=train_transforms )
    n = len(dataset)
    split=int(n*0.8)
    indices = list(range(n))
    random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
  
    train_set = torch.utils.data.Subset(dataset, train_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=128)
    test_loader = DataLoader(test_set, batch_size=128)

    mean, std = get_mean_std(train_loader)

    data_transforms = transforms.Compose([
        transforms.Resize((180, 180)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)    
        ])
    
    dataset = ImageFolder(root="flower_photos", transform=data_transforms )
    n = len(dataset)
    split=int(n*0.8)
    indices = list(range(n))
    random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
  
    train_set = torch.utils.data.Subset(dataset, train_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=128)
    test_loader = DataLoader(test_set, batch_size=128)



    #Create model and get summary
    model = Net()
    print(model)
    model.share_memory()

    #Define loss function and optimizer
    loss_fn= nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    processes=[]

    for rank in range(workers):
        p=Process(target=train, args=(train_loader, model, loss_fn, opt, epochs, workers))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    PATH = './flowers_net.pth'
    torch.save(model.state_dict(), PATH)

    model = Net()
    model.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs=model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total} %')
    print(f'Epochs: {epochs}')
    print(f'Elapsed time : {time.time()-t0}')
    print(f'Workers: {workers}')
    

