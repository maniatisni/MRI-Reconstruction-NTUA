# %%
import os
import time
from numpy import DataSource
import torch
import torch.nn as nn
from torch.nn.modules.loss import MSELoss
from functions import *
from functions import ReshComplex_Img
from torch.autograd import Variable

# from AllConversionsNeeded import * 
# import torch.optim as optim
# from torch.autograd import Variable
# import argparse
# import matplotlib.pyplot as plt
# %%
relu = nn.ReLU()
class simple_net(nn.Module):
    """
    Input on the net is [batch_size, 256, 256, 2]
    Output we want is [batch_size, 256, 256, 2]
    """
    def __init__(self): 
        super(simple_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels=2, kernel_size=3, padding = 1)
    def forward(self,x):
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = relu(x)
        x = self.conv3(x)
        x = relu(x)
        return x


# %% 
DATA_PATH = '/home/nick/Biodata/DATA/'
ImagesVal = DataLoader(DATA_PATH + 'VAL/')
# %% 
ImagesVal = ImagesVal[0:5,:,:,:]
# ImagesTest = DataLoader(DATA_PATH + 'TEST/')
##################################################################
# Create the Nufft Objects
nufft_ob, adjnufft_ob =CreateNuffts()
##################################################################
# Bring Image in TorchObject Format (#,1,1,256,256)
imgs_PyFormat_Val= ImgCreate(ImagesVal)
print( 'imgs_PyFormat_Val shape', imgs_PyFormat_Val.shape)
##################################################################
# Create Radial Trajectory 
ktraj, grid_size, nspokes, nsamples  =  Rad2DTraj()
ktraj = torch.tensor(ktraj).to(torch.float)
print(ktraj.shape)
##################################################################
# Create the Radial k-Space Data (#,1,1,65536)
kdata_Val = CalcRad_kSpace_All(imgs_PyFormat_Val, ktraj)
print('kdata_Val shape:',kdata_Val.shape)
##################################################################
# Create Radial Undersampling Mask (#1,1,65536)
Val_Masks = AllMasks(kdata_Val)
print('Val_Masks shape:', Val_Masks.shape)
##################################################################
# Create Undersmpled Radial kSpace Data ((#,65536) or(#,256,256))
us_kdata_Val = kSpaceR_US(kdata_Val, Val_Masks, res =False)
print('us_kdata_Val shape:', us_kdata_Val.shape)
##################################################################
# Create Undersampled Reconstructed Images ((#,256,256))
us_Img_Val = RecImgAll(us_kdata_Val,ktraj)
print('us_Img_Val shape:', us_Img_Val.shape)
plt.imshow(np.abs(us_Img_Val[0,:,:]))
# %%

# %% 
num_epoch = 50
model = simple_net()
criterion = MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
device = torch.device("cuda")
train = ReshComplex_Img(us_Img_Val)
# labels = ReshComplex_Img(ImagesVal)
if cuda:
    model = model.cuda()
    criterion.cuda()

i = 0
for epoch in range(num_epoch):
    t_start = time.time()
    # Training
    train_err = 0
    train_batches = 0
    for im in range(us_Img_Val.shape[0]):
        train_sample = train[im,:,:]
        ground_truth = ImagesVal[im,:,:]

        train_sample = torch.from_numpy(train_sample).to(device,dtype=torch.float)
        ground_truth = torch.from_numpy(ground_truth).to(device,dtype=torch.float)
        train_sample = torch.reshape(train_sample,(2,256,256))
        ground_truth = torch.reshape(ground_truth,(2,256,256))
        optimizer.zero_grad()
        model.zero_grad()
        outputs = model(train_sample[None,...])
        loss = criterion(outputs, ground_truth)
        loss.backward()
        optimizer.step()
        train_err += loss.item()
    print('Epoch #{} and Loss: {}'.format(epoch, train_err))
# %%
