# resnet_resnext

This is a simple code for training resnet / resnext using FashionMNIST dataset.

---
## Experimental setting
1. learning rate : 3e-4
2. Epoch : 50
3. Batch size : 48
4. Model depth : 50
5. Types of optimizer : Adam

## Experimental results
|Model|Flops|#params|Accuracy(top1)|Accuracy(top5)|
|-----|-----|-------|--------------|--------------|
|ResNet|2.64G|23.52M|92.80%|99.73%|
|ResNext|2.74G|22.99M|92.83%|9.79%|
