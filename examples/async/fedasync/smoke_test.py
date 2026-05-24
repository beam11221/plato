# Standalone script. No PLATO. No FL. Just torch.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Your GN VGG model — import or paste exactly the class you're using in PLATO
from plato.models.vgg import Model

model = Model.get('vgg_11').cuda()

# CIFAR-10
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5,)*3, (0.5,)*3)])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=256, num_workers=2)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    losses = []
    for x, y in trainloader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    # Eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.cuda(), y.cuda()
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"Epoch {epoch}: train_loss={sum(losses)/len(losses):.4f}, test_acc={correct/total:.4f}")