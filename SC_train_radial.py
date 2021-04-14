# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 15:01:30 2021

@author: WScott
"""


import matplotlib.pylab as plt
import numpy as np
import os
import glob
import sys
import random 
import time 
import h5py

# Importing our model
#MY_UTILS_PATH = "../Modules/"
#if not MY_UTILS_PATH in sys.path:
#    sys.path.append(MY_UTILS_PATH)
import Modules.cs_models_sc as fsnet

from Modules.All_Radial_Functions import *


# Importing callbacks and data augmentation utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import  Adam

import tensorflow as tf

# %% PARAMETERS
H,W = 256,256 # Training image dimensions
channels = 2 # complex data 0-> real; 1-> imaginary
norm = np.sqrt(H*W)

# Train Set 
#train_path = "/home/ubuntu/volume1/Raw-data/SC/Train/*.npy"
#train_path = '../input/calgary-campinas/*.npy'

train_path = 'D:/DIMITRIS_2021/DATA/TRAIN_ALL/*.npy'

kspace_files_train = np.asarray(glob.glob(train_path))

# Validation set
#val_path = "/home/ubuntu/volume1/Raw-data/SC/Val/*.npy"
#val_path = '../input/calgarycampinas-val/*.npy'
val_path ='D:/DIMITRIS_2021/DATA/Single-channel_a/Val/*.npy'
kspace_files_val = np.asarray(glob.glob(val_path))

indexes = np.arange(kspace_files_train.size,dtype = int)
np.random.shuffle(indexes)
kspace_files_train = kspace_files_train[indexes]


print(kspace_files_train[-1])
print(len(kspace_files_train)) 

print(kspace_files_val[-1])
print(len(kspace_files_val))

# %% Training Data 
# Get number of samples
ntrain = 0
#for ii in range(len(kspace_files_train)):
for ii in range(0,1):
    ntrain += np.load(kspace_files_train[ii]).shape[0] 

# Load train data    
rec_train = np.zeros((ntrain,H,W,2))

aux_counter = 0 
#for ii in range(len(kspace_files_train)):
for ii in range(0,1):
     aux_kspace = np.load(kspace_files_train[ii])/norm
     aux = int(aux_kspace.shape[0])
     aux2 = np.fft.ifft2(aux_kspace[:,:,:,0]+\
                         1j*aux_kspace[:,:,:,1])
     rec_train[aux_counter:aux_counter+aux,:,:,0] = aux2.real
     rec_train[aux_counter:aux_counter+aux,:,:,1] = aux2.imag
     aux_counter+=aux

# Shuffle training    
indexes = np.arange(rec_train.shape[0],dtype = int)
np.random.shuffle(indexes)
rec_train = rec_train[indexes]
print("Number of training samples", rec_train.shape[0])


# %% Validataion Data 
# Get number of samples
nval = 0
#for ii in range(len(kspace_files_val)):
for ii in range(0,1):
    nval += (np.load(kspace_files_val[ii]).shape[0] )

# Load Validation data
kspace_val = np.zeros((nval,H,W,2))
rec_val = np.zeros((nval,H,W,2))
aux_counter = 0
#for ii in range(len(kspace_files_val)):
for ii in range(0,1):
    aux_kspace = np.load(kspace_files_val[ii])/norm
    aux = int(aux_kspace.shape[0])
    kspace_val[aux_counter:aux_counter+aux] = aux_kspace
    aux2 = np.fft.ifft2(aux_kspace[:,:,:,0]+1j*aux_kspace[:,:,:,1])
    rec_val[aux_counter:aux_counter+aux,:,:,0] = aux2.real
    rec_val[aux_counter:aux_counter+aux,:,:,1] = aux2.imag
    aux_counter+=aux

print("Number of val samples", kspace_val.shape[0])

#%% Some Variables needed for the following Calculations
H,W = rec_val.shape[1], rec_val.shape[2]
norm = np.sqrt(H*W)
im_size = (H,W)
print( 'H, W:', H, W)

#%% 
os.chdir('D:/DIMITRIS_2021/tfkbnufft_master') 
from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.kbnufft import KbNufftModule
from tfkbnufft.mri.dcomp_calc import calculate_radial_dcomp_tf, calculate_density_compensator
dtype = tf.float32

#%% Image Create  !!
#Convert Data to the neded format 

# rec_train_Im = ImgCreate(rec_train) #[3274,1,1,256,256]
rec_val_Im = ImgCreate(rec_val) #[1700,1,1,256,256]
print( 'rec_val_Im shape', rec_val_Im.shape)

#%% Calculate the Radial 2D Trajectory !!
# Create Radial Trajectory 

ktraj, grid_size, nspokes, spokelength = Rad2DTraj(nspokes=256, ga=True)
# convert k-space trajectory to a tensor and unsqueeze batch dimension
ktraj = tf.convert_to_tensor(ktraj)[None, ...]
print('ktraj shape: {}'.format(ktraj.shape))

#%% Create TFkbNufft Object 

# create NUFFT objects, use 'ortho' for orthogonal FFTs
nufft_ob = KbNufftModule(im_size=im_size, grid_size=grid_size, norm='ortho')
print(nufft_ob)

#%% Create Radial kSpace 
#rec_val_Im2 = rec_val_Im[100:102,:,:,:,:]
#kspace_val_Rad2 = CD_kSpaceRad(rec_val_Im2, ktraj)

print( 'rec_val_Im shape', rec_val_Im.shape)
kspace_val_Rad = CD_kSpaceRad(rec_val_Im, ktraj)
#kspace_val_Rad = CD_kSpaceRad(rec_val, ktraj)


#%%  Create val_var_sampling_mask_R

image1 = rec_val_Im[100,:,:,:,:]
kldata = CalcRad_kSpace(image1,ktraj)

val_var_sampling_mask_R = np.zeros((kspace_val_Rad.shape[0],H,W,2), dtype=np.int64)

#for i in range(0,100):
for i in range(0,kspace_val_Rad.shape[0]):
    mask , defOfUnd = Rad2DUndMask(kldata,256,256,und=0.8)
    val_var_sampling_mask_R[i,:,:,:] = mask
print('val_var_sampling_mask_R:', val_var_sampling_mask_R.shape)

plt.imshow(val_var_sampling_mask_R[30,:,:,0])


#%% Test that the recon is working 
# Example to run the recon for all 
img = AdjNufftBack(kspace_val_R2, ktraj, method=3)
    
#%% Create an hdf5 file to save Model weights 
arr = []

with h5py.File('file_20210409.hdf5', 'w') as f:
    dset = f.create_dataset("default", data=arr)
#%% Model Details 

# LOAD the file to SAVE the TRAINED MODEL 
epochs = 10
batch_size= 6

#model_name = "../Models/flat_unrolled_cascade_ikikii.hdf5"
#######################################################################################
#model_name = '../input/cd-deepcascade-sc-savemodel/myModels.hdf5'
model_name = 'D:/DIMITRIS_2021/Modules/file_20210408.hdf5'

# Early stopping callback to shut down training after
# 5 epochs with no improvement
earlyStopping = EarlyStopping(monitor='val_loss',
                                       patience=20, 
                                       verbose=0, mode='min')

# Checkpoint callback to save model  along the epochs
checkpoint = ModelCheckpoint(model_name, mode = 'min', \
                         monitor='val_loss',verbose=0,\
                         save_best_only=True, save_weights_only = True)
    
#%% Data Augmentation for radial Imaging 

def combine_generator_R(gen1, gen2, under_masks):
    ''' This Generator Creates Radial Data and Mask them for the training dataset'''
    while True:
        rec_real = gen1.next()
        rec_imag = gen2.next()
        
        rec_complex_R = np.zeros((rec_real.shape[0],rec_real.shape[1],rec_real.shape[2],2))
        rec_complex_R[:,:,:,0] = np.squeeze(rec_real)
        rec_complex_R[:,:,:,1] = np.squeeze(rec_imag) 
        rec_train_Im = ImgCreate(rec_complex_R) # Brings image in the rig
        
        kspace2_R = CD_kSpaceRad(rec_train_Im, ktraj)
        indexes = np.random.choice(np.arange(under_masks.shape[0], dtype=int), rec_real.shape[0], replace=False)
        kspace2_R[under_masks[indexes]] = 0
        
        yield([kspace2_R,under_masks[indexes].astype(np.float32)],[rec_complex_R])
        
        
        
seed = 905
image_datagen1 = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

image_datagen2 = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')


image_generator1 = image_datagen1.flow(rec_train [:,:,:,0,np.newaxis],batch_size = batch_size,seed = seed)
image_generator2 = image_datagen2.flow(rec_train[:,:,:,1,np.newaxis],batch_size = batch_size,seed = seed)        


#combined = combine_generator(image_generator1,image_generator2, var_sampling_mask)
combined_R = combine_generator_R(image_generator1,image_generator2, val_var_sampling_mask_R)
 
    
    
    
#%% Crete Undersampling Validation - Radial Data 

print('RADIAL:','\n kspace:', kspace_val_Rad.shape, '\n sampling mask:',val_var_sampling_mask_R.shape, '\n image(rec_val):', rec_val.shape)

#%% 
indexes = np.random.choice(np.arange(val_var_sampling_mask_R.shape[0],dtype =int),kspace_val_Rad.shape[0],replace = True)
val_var_sampling_mask_R = (val_var_sampling_mask_R[indexes]) #(1700, 256,256,2) ValSize
val_var_sampling_mask_R = np.asarray(val_var_sampling_mask_R, dtype=bool)

# Create a list with kSpace and Sampling Masks 
lst_R=[]
for i in range(0, len(kspace_val_Rad)):
    lst_R.append([kspace_val_Rad[i,:,:,:],val_var_sampling_mask_R[i,:,:,:]])
print('Done')

#%% Function to make Zero all the entries 

def rstKspace_R(args):
    kspace_val=args[0]
    val_var_sampling_mask = args[1]
    kspace_val[val_var_sampling_mask] = 0
    out_R = kspace_val
    return out_R

#%% Mask kSpace 

kSpace=[]
for i in range(0, len(kspace_val_Rad)):
    kSpace.append(rstKspace_R(lst_R[i]))
    
#Convert to array from list 
kspace_val_R = np.asarray(kSpace)

print('kspace_val_R:',kspace_val_R.shape)


#%% Train the model 

model = fsnet.deep_cascade_flat_unrolled_R("ikikii", H, W)
opt = Adam(lr = 1e-3,decay = 1e-4)
model.compile(loss = 'mse',optimizer=opt)
print(model.summary())

#%% Run the model 
#steps_per_epoch:  value as the total number of training data points divided by the batch size
hist = model.fit_generator(combined_R,
             epochs=epochs,
             steps_per_epoch=rec_train.shape[0]//batch_size, # Needed for the ImageDataGenerator not to run for ever
             verbose=1,
             validation_data= ([kspace_val_R,val_var_sampling_mask_R],[rec_val]),
             callbacks=[checkpoint,earlyStopping])

#print('RADIAL:','\n kspace:', kspace_val_Rad.shape, '\n sampling mask:',val_var_sampling_mask_R.shape, '\n image(rec_val):', rec_val.shape)
