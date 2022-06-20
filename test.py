import os
import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from ptflops import get_model_complexity_info
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

transform=transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(), transforms.Normalize( (0.5,), (0.5,))])

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

class MnistResNet(nn.Module):
  def __init__(self, in_channels=1):
    super(MnistResNet, self).__init__()

    # Load a pretrained resnet model from torchvision.models in Pytorch
    # self.model = models.resnet50(pretrained=False)
    self.model = torchvision.models.resnext50_32x4d()

    # Change the input layer to take Grayscale image, instead of RGB images. 
    # Hence in_channels is set as 1 or 3 respectively
    # original definition of the first layer on the ResNet class
    # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Change the output layer to output 10 classes instead of 1000 classes
    num_ftrs = self.model.fc.in_features
    self.model.fc = nn.Linear(num_ftrs, 10)

  def forward(self, x):
    return self.model(x)


model = MnistResNet().to(DEVICE)

dummy_size = (1, 256, 256)

macs, params = get_model_complexity_info(model, dummy_size, as_strings=True, print_per_layer_stat=True, verbose=True)
                                
# print('computational complexity: ', macs)
# print('number of parameters: ', params)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

model.eval()
top_1 = 0
top_5 = 0

model_state_dict = torch.load("ResNext50.pth")
model.load_state_dict(model_state_dict)

for i, (data, target) in enumerate(test_loader) :
    with torch.no_grad() :
        target = target.type(torch.LongTensor)
        data, target = data.to(DEVICE), target.to(DEVICE)
        outputs = model(data)

        res = accuracy(outputs, target, (1, 5))
        top_1 += res[0]
        top_5 += res[1]
        print('{}/{} {} {}'.format(i+1, len(test_loader), res[0], res[1]))
print(top_1/len(test_loader), top_5/len(test_loader))
print('computational complexity: ', macs)
print('number of parameters: ', params)