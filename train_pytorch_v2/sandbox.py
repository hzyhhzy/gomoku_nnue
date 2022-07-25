import numpy as np
import torch
import torch.nn as nn
import time
import torch.optim as optim

a=torch.tensor(list(range(3*4*5))).reshape(3,4,5)
b=[1,2,1]
print(a[b])
exit(0)


a=nn.Conv2d(114,514,(1,9),groups=2)
print(a.weight.shape)


class Model1(nn.Module):

    def __init__(self,c=256,b=5):
        super().__init__()
        self.conv0=nn.Conv2d(1,c,1)
        self.trunk=nn.ModuleList()
        for i in range(b):
            self.trunk.append(nn.Conv2d(c,c,1))

        self.conv1 = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        x=self.conv0(x)
        for block in self.trunk:
            x=block(x)

        x = self.conv1(x)
        x = x.mean(axis=(0,1,2,3))
        return x

class Model2(nn.Module):

    def __init__(self, c=256,b=5):
        super().__init__()
        self.conv0 = nn.Linear(1, c)
        self.trunk = nn.ModuleList()
        for i in range(b):
            self.trunk.append(nn.Linear(c, c))

        self.conv1 = nn.Linear(c, 1)

    def forward(self, x):
        x = self.conv0(x)
        for block in self.trunk:
            x = block(x)

        x = self.conv1(x)
        x = x.mean(axis=(0,1))
        return x

device=torch.device("cuda:0")
a = Model1(256,20).to(device)
b = Model2(256,20).to(device)

aoptimizer = optim.Adam(a.parameters(), lr=1e-6)
boptimizer = optim.Adam(b.parameters(), lr=1e-6)


time0 = time.time()
for step in range(1000):
    inp=torch.tensor(np.random.normal(size=(256,1,15,15)),dtype=torch.float32).to(device)
    loss=a(inp)
    loss.backward()
    aoptimizer.step()

time1 = time.time()
print(time1-time0)


for step in range(1000):
    inp=torch.tensor(np.random.normal(size=(256*15*15,1)),dtype=torch.float32).to(device)
    loss=b(inp)
    loss.backward()
    boptimizer.step()

time2 = time.time()
print(time2 - time1)