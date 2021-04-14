# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 11:52:29 2021

@author: DTryfonopoulos
"""
#%matplotlib inline
import matplotlib.pylab as plt
import numpy as np
import os
import glob
import sys

#%% Importing our model
MY_UTILS_PATH ='D:\DIMITRIS_2021\Modules'
#MY_UTILS_PATH = "../Modules/"
if not MY_UTILS_PATH in sys.path:
    sys.path.append(MY_UTILS_PATH)
import cs_models_sc as fsnet

# Importing callbacks and data augmentation utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import  Adam

#%% 
## PARAMETERS
H,W = 256,256 # Training image dimensions

channels = 2 # complex data 0-> real; 1-> imaginary
norm = np.sqrt(H*W)


test_path = "D:\DIMITRIS_2021\DATA\Single-channel_a\Test\*.npy"
kspace_files_test = np.asarray(glob.glob(test_path))


print(kspace_files_test[-1])
print(len(kspace_files_test))

#%%
def Poisson(dim, und=0.2):
    import numpy as np 
    import numpy.matlib
    import random 
    A = np.arange(0,dim,1)
    B = np.matlib.repmat(A,dim,1)
    C=np.repeat(B[np.newaxis,:, :], 100, axis=0)
    
    for i in range(np.size(C,0)):
        for j in range(np.size(C,1)):
            C[i,j,np.array(random.sample(range(0, dim), int(dim*und)))] =0
    C[C>0]=1
    C[:,int(dim/2)-int(dim/2*0.1):int(dim/2)+int(dim/2*0.1),int(dim/2)-int(dim/2*0.1):int(dim/2)+int(dim/2*0.1)]=1
    C[C==0]=False
    C[C==1]=True
    und_mask =C
    return und_mask

# %% [code]
C = Poisson(256)
var_sampling_mask = (np.fft.fftshift(C, axes=(1,2)))

#%%
var_sampling_mask = np.concatenate((var_sampling_mask[:,:,:,np.newaxis],var_sampling_mask[:,:,:,np.newaxis]),\
                                          axis = -1)


# White pixels are retrospectively discarded
plt.figure(dpi = 250)
for ii in range(5):
    plt.subplot(1,5,ii+1)
    plt.imshow(var_sampling_mask[ii*10,:,:,0],cmap = "gray")
    plt.axis("off")
plt.show()

print("Undersampling:", 1.0*var_sampling_mask.sum()/var_sampling_mask.size)


#%%
# Loading sampling patterns. Notice that here we are using uncentred k-space
# var_sampling_mask = np.fft.fftshift(~np.load("../Data/Sampling-patterns/256x256/poisson_center_radius=18_20perc.npy") \
  #                                  ,axes = (1,2))
#var_sampling_mask = np.concatenate((var_sampling_mask[:,:,:,np.newaxis],var_sampling_mask[:,:,:,np.newaxis]),\
#                                         axis = -1)[0]

var_sampling_mask = np.fft.fftshift(Poisson(256,0.8))
var_sampling_mask = np.concatenate((var_sampling_mask[:,:,:,np.newaxis],var_sampling_mask[:,:,:,np.newaxis]),\
                                          axis = -1)


# White pixels are retrospectively discarded
plt.figure()
plt.imshow(var_sampling_mask[:,:,0],cmap = "gray")
plt.axis("off")
plt.show()

print("Undersampling:", 1.0*var_sampling_mask.sum()/var_sampling_mask.size)

#%% Training our model
model_name = "D:/DIMITRIS_2021/mytestfile.hdf5"
model = fsnet.deep_cascade_flat_unrolled("ikikii", H, W)
opt = Adam(lr = 1e-3,decay = 1e-5)
model.compile(loss = 'mse',optimizer=opt)
model.load_weights(model_name)

#%%
si = 100 # slice to display
for ii in range(len(kspace_files_test)):
for ii in range(0,1):
    kspace_test = np.load(kspace_files_test[ii])/norm
    rec_test = np.zeros(kspace_test.shape)
    aux = np.fft.ifft2(kspace_test[:,:,:,0]+1j*kspace_test[:,:,:,1])
    rec_test[:,:,:,0] = aux.real
    rec_test[:,:,:,1] = aux.imag
    var_sampling_mask_test = np.tile(var_sampling_mask,(kspace_test.shape[0],1,1,1))
    kspace_test[:,var_sampling_mask] = 0
    pred = model.predict([kspace_test,var_sampling_mask_test])
    #name = kspace_files_test[ii].split("/")[-1].split(".npy")[0]
    #np.save(name + "_rec.npy",pred)
    plt.figure()
    plt.subplot(121)
    plt.imshow(np.abs(pred[si,:,:,0]+1j*pred[si,:,:,1]),cmap = "gray")
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(np.abs(rec_test[si,:,:,0]+1j*rec_test[si,:,:,1]),cmap = "gray")
    plt.axis("off")
    plt.show()