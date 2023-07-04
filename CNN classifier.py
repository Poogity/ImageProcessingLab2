import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters
input_size = 784 #28x28
num_classes = 10
num_epochs = 5
b = 50 #batch size
learning_rate = 0.1
trainingCostArray = []
testingCostArray = []
accuracyArray = []
confusionMatrix = np.empty((10,10))
train_dataset = torchvision.datasets.MNIST(root='./data',train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size = b,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset= test_dataset, batch_size = b, shuffle=False)

classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')

examples = iter(train_loader)
samples, labels = examples.next()
for i in range(10):
    j=0
    plt.subplot(2, 5, i+1)
    while labels[j]!=i:
        j+=1
    plt.imshow(samples[j][0], cmap='gray'), plt.xticks([]), plt.yticks([]) 
plt.show()

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        
        self.convolutional_layer = nn.Sequential(
            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1),
            nn.ReLU(),     
            nn.BatchNorm2d(6),
            
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=216, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10)
        )
        
    def forward(self, x):
        x = self.convolutional_layer(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)
        x = F.softmax(x, dim=1)
        return x

def test(model):
    #----------------------Testing----------------------------------------
    loss = []
    confusionMatrix = np.zeros((10,10))
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            loss.append(criterion(outputs, labels))
            for i in range(b):
                label = labels[i]
                pred = predicted[i]
                
                confusionMatrix[label][pred]+= float(1)
                
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1
                
        acc = 100.0 * n_correct / n_samples
        accuracyArray.append(acc)
        avgLoss = np.mean(loss)
        testingCostArray.append(avgLoss)
        print(f'Accuracy of the network: {acc} %')
    
        for i in range(10):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc} %')   
            
            confusionMatrix[i]/=n_class_samples[i]
confusionMatrix = confusionMatrix.T
model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#----------------------Training----------------------------------------
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
       
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        trainingCostArray.append(loss.detach().numpy())
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 200 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
    test(model)
x = np.linspace(0,num_epochs,1200*num_epochs)
plt.title("Loss during training")
plt.xlabel("epoch")
plt.ylabel("Cross Entropy Loss")
plt.plot(x,trainingCostArray)
plt.show()

x = np.linspace(1,num_epochs,num_epochs)
plt.title("Accuracy after every epoch")
plt.xlabel("epoch")
plt.ylabel("accuracy (%)")
plt.plot(x,accuracyArray)
plt.show()

plt.xlabel("epoch")
plt.ylabel("CE Loss")
plt.title("Average Loss after every epoch")
plt.plot(x,testingCostArray)
print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)