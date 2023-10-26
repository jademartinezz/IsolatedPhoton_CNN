'''
This is the "All in one File" for the fourth version of th
e PyTorch code for the "Machine Learning for Isolated Phot
on Identification" project. These files attempt to identif
y isolated photons through plotting the tracks as p_T asym
metry vs. ∆R tracks and  visualizing them as boxes instead
of contour track plots. This model is built using 1000 plo
ts rather than 200 plots. This file contains the following
files: Datasets & data loaders, building the neural networ
k, and optimization.
8/1/2023
Jade Martinez
'''
#%% ------------------------- DATASETS & DATALOADERS ------------------------------
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor, Lambda
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# applying transformations
# tuples inside normalize function should be mean(1st) and standard dev(2nd)
transform = transforms.Compose(
    [transforms.ToTensor(), # converts array into float tensor
    transforms.Normalize((0.4,0.4,0.4),(0.2,0.2,0.3))] # normalize to speed up learning
    )
target_transform = Lambda(lambda y: torch.zeros(
    24, dtype=torch.float32).scatter_(dim=0, index=torch.tensor(y), value=1))


# Creating custom data set
class IsoPhotTrainData(Dataset):
    # initialize directory, file & transformations
    def __init__(self, IMG_LABELS, IMG_DIR, transform=None, target_transform=None):
        self.imgLabels = pd.read_csv(IMG_LABELS)
        self.imgDir = IMG_DIR
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64,64)),
            transforms.ToTensor()
        ])
        self.target_transform = target_transform
        
    # number of samples in dataset
    def __len__(self):
        return len(self.imgLabels)
    
    # returns sample from dataset (image and corresponding label)
    def __getitem__(self, indx):
        imgPath = os.path.join(self.imgDir, self.imgLabels.iloc[indx,0])
        image = read_image(imgPath)
        label = self.imgLabels.iloc[indx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
# loading a dataset
trainingData = IsoPhotTrainData(
    'data/train/IsoPhotIDlabels.csv',
    'data/train/photJPEGS'
)

testData = IsoPhotTrainData(
    'data/test/IsoPhotIDlabels.csv',
    'data/test/photJPEGS'
)

# prep data for training
trainDataLoader = DataLoader(trainingData, batch_size = 24, shuffle = 1)
testDataLoader = DataLoader(testData, batch_size = 24, shuffle = 1) # can include num_workers

# iterating and visualizing the dataset
labelsTable = {0: "Isolated photon", 
               1: "Photon from jet"}
figure = plt.figure(figsize=(8,8))
cols, rows = 3,3
for i in range (1, cols*rows + 1):
    sampleIndex = torch.randint(len(trainingData), size=(1,)).item()
    img, label = trainingData[sampleIndex]
    figure.add_subplot(rows, cols, i)
    plt.title(labelsTable[label])
    plt.axis("off")
    plt.imshow(img.permute(1,2,0)) # can add cmap feature
plt.show()

# iterating through the data loader
trainFeat, trainLabels = next(iter(trainDataLoader))

'''
UNCOMMENT TO PRINT
print(f'Feature batch shape: {trainFeat.size()}')
print(f'Labels batch shape: {trainLabels.size()}')
'''

img = trainFeat[0].squeeze()
label = trainLabels[0]
plt.imshow(img.permute(1,2,0)) # can include cmap
plt.show()
print(f'Label: {label}')
print ('------------- DATASETS & DATALOADERS COMPLETED -------------')

#%% --------------------- BUILDING THE NEURAL NETWORK -----------------------
import torch
from torch import nn
from torch.utils.data import DataLoader

# get device for training
device = ("cpu") # can use gpu or mps

# defining the class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__() # self() gives access to parent methods
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(12288, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
            )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# create an instance of the net and move it to the device
model = NeuralNetwork().to(device)

# getting prediction probabilities
X = torch.rand(24, 64,3* 64, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred1 = pred_probab.argmax(1)
y_pred = y_pred1.type('torch.FloatTensor')


#print(f"Predicted class: {y_pred}")
#print(X.dtype,logits.dtype, pred_probab.dtype, y_pred.dtype)

print('----------------- NEURAL NETWORK COMPLETED -----------------')

#%% ------------------------ OPTIMIZATION ----------------------------------
learning_rate = 1e-4
batch_size = 24
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        pred = pred.type('torch.FloatTensor')
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float32).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 80
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(trainDataLoader, model, loss_fn, optimizer)
    test_loop(testDataLoader, model, loss_fn)
print("Done!")

# Checking on test data 
testData2 = IsoPhotTrainData(
    'data/onlyTest/IsoPhotIDlabels.csv',
    'data/onlyTest/photJPEGS'
)

testDataLoader2 = DataLoader(testData2, batch_size = 197, shuffle = 1)
