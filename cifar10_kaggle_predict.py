# Training an image classifier using CNN (CIFAR10 dataset)

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import xlsxwriter
import glob
from PIL import Image


batch_size = 4
data_dir = '/media/anand/ce9cd9fd-01b4-40bc-8862-3282a6e2471a/Kanmani/cifar-10-full-dataset/train/'
data_dir1 = '/media/anand/ce9cd9fd-01b4-40bc-8862-3282a6e2471a/Kanmani/cifar-10-full-dataset/test/'

transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



def load_split_train_test(datadir, valid_size = .2):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = torchvision.datasets.ImageFolder(datadir, transform=transform)
    test_data = torchvision.datasets.ImageFolder(datadir, transform=transform)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=batch_size, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=batch_size, num_workers=2)
    return trainloader, testloader


trainloader, testloader = load_split_train_test(data_dir, .2)

predict_data = torchvision.datasets.ImageFolder(data_dir1, transform=transform)
predictloader = torch.utils.data.DataLoader(predict_data, batch_size=1, num_workers=2)

print('trainloader', len(trainloader))
print('testloader', len(testloader))
print('predictloader', len(predictloader))

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Show some images for fun

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))



# Define a CNN

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


# Define a loss function and optimizer

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)


# Train the network

for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 1000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

print('Finished Training')


# Save our model

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)


# Test the network on the test data
'''
# Test it for 4 random test images

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))



net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)


_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
'''

# Check how the network performs on the whole dataset

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# Analyze - Classes that performed well, Classes that did not perform well

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
count = 0
with torch.no_grad():
    for data in testloader:
        count = count + 1
        images, labels = data

        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


# Predict the images and write it in an excel file.

# Start from the first cell. Rows and columns are zero indexed.
row = 1
col = 0
col1 = 1

# Iterate over the data and write it out row by row.

workbook = xlsxwriter.Workbook('/media/anand/ce9cd9fd-01b4-40bc-8862-3282a6e2471a/Kanmani/pythonProject/sampleSubmission.xlsx')
print('create xl')
worksheet = workbook.add_worksheet()


# Read Images from the folder, predict the class and write it into an Excel file

names=glob.glob("/media/anand/ce9cd9fd-01b4-40bc-8862-3282a6e2471a/Kanmani/cifar-10-full-dataset/test/1/*.png")
for f in names:
    image = Image.open(f)
    image_2_npArray = np.asarray(image)
    image_2_npArray_2_tensor = torchvision.transforms.ToTensor()(image_2_npArray)
    image_2_npArray_2_tensor = image_2_npArray_2_tensor.unsqueeze(0)
    outputs = net(image_2_npArray_2_tensor)
    _, predicted = torch.max(outputs.data, 1)
    result = f.find('test/1')
    result = result + len('test/1/')
    data = f[result:]
    data = data[:-4]
    # print('predicted', predicted)
    worksheet.write(0, col, 'id')
    worksheet.write(0, col1, 'Label')
    worksheet.write(row, col, data)
    worksheet.write(row, col1, classes[predicted])
    row += 1

workbook.close()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)
net.to(device)