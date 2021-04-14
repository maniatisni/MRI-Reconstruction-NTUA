# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:35:11 2021

@author: DTryfonopoulos
"""
#%matplotlib inline
import matplotlib.pylab as plt
import numpy as np
import os
import glob
import sys

# Importing our model
#MY_UTILS_PATH ='C:/Users/DTryfonopoulos/OneDrive - MR Solutions/Documents/PhD/MARKO-PIZURICA_Project/CD-Deep-Cascade-MR-Reconstruction-master/CD-Deep-Cascade-MR-Reconstruction-master/Modules'
#MY_UTILS_PATH = "../Modules/"
#if not MY_UTILS_PATH in sys.path:
 #   sys.path.append(MY_UTILS_PATH)

import cs_models_sc as fsnet

# Importing callbacks and data augmentation utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import  Adam


#%% PARAMETERS 

H,W = 256,256 # Training image dimensions


channels = 2 # complex data 0-> real; 1-> imaginary
norm = np.sqrt(H*W)

# Train Set 
train_path = 'C:/Users/DTryfonopoulos/Documents/conp-dataset/projects/calgary-campinas/.git/annex/objects/train_all/*.npy'
#train_path = "/home/ubuntu/volume1/Raw-data/SC/Train/*.npy"
kspace_files_train = np.asarray(glob.glob(train_path))

# Validation set
val_path = "C:/Users/DTryfonopoulos/Documents/conp-dataset/projects/calgary-campinas/.git/annex/objects/val_all/*.npy"
kspace_files_val = np.asarray(glob.glob(val_path))

indexes = np.arange(kspace_files_train.size,dtype = int)
np.random.shuffle(indexes)
kspace_files_train = kspace_files_train[indexes]


print(kspace_files_train[-1])
print(len(kspace_files_train))

print(kspace_files_val[-1])
print(len(kspace_files_val))

#%%  GET # of SAMPLES  TRAINING
ntrain = 0
for ii in range(len(kspace_files_train)):
    ntrain += np.load(kspace_files_train[ii]).shape[0] 

# Load train data    
rec_train = np.zeros((ntrain,H,W,2))

aux_counter = 0
for ii in range(len(kspace_files_train)):
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

#%%   Get number of samples VALIDATION
nval = 0
for ii in range(len(kspace_files_val)):
    nval += (np.load(kspace_files_val[ii]).shape[0] )

# Load Validation data
kspace_val = np.zeros((nval,H,W,2))
rec_val = np.zeros((nval,H,W,2))
aux_counter = 0
for ii in range(len(kspace_files_val)):
    aux_kspace = np.load(kspace_files_val[ii])/norm
    aux = int(aux_kspace.shape[0])
    kspace_val[aux_counter:aux_counter+aux] = aux_kspace
    aux2 = np.fft.ifft2(aux_kspace[:,:,:,0]+1j*aux_kspace[:,:,:,1])
    rec_val[aux_counter:aux_counter+aux,:,:,0] = aux2.real
    rec_val[aux_counter:aux_counter+aux,:,:,1] = aux2.imag
    aux_counter+=aux

print("Number of samples", kspace_val.shape[0])
#%%
def Poisson(dim, und=0.75):
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

#%% LOADING SAMPLING PATTERNS - Uncentered kSpace 
# Loading sampling patterns. Notice that here we are using uncentred k-space
#var_sampling_mask = np.fft.fftshift(~np.load("../Data/Sampling-patterns/256x256/poisson_center_radius=18_20perc.npy") \
 #                                   ,axes = (1,2))

#var_sampling_mask = np.fft.fftshift(np.fft.fftshift(~np.load("C:/Users/DTryfonopoulos/OneDrive - MR Solutions/Documents/PhD/MARKO-PIZURICA_Project/CD-Deep-Cascade-MR-Reconstruction-master/CD-Deep-Cascade-MR-Reconstruction-master/Data/poisson_sampling/R10_218x170.npy") \
                                   # ,axes = (1,2)))

#var_sampling_mask = np.resize(var_sampling_mask, (100, 256,256))

#var_sampling_mask = np.fft.fftshift(Poisson(256,0.8))
var_sampling_mask = np.fft.fftshift(Poisson(256,0.8))
var_sampling_mask = np.concatenate((var_sampling_mask[:,:,:,np.newaxis],var_sampling_mask[:,:,:,np.newaxis]),\
                                          axis = -1)

# ORIGINAL version
#var_sampling_mask = np.fft.fftshift(~np.load("../Data/Sampling-patterns/256x256/poisson_center_radius=18_20perc.npy") \
                                   # ,axes = (1,2))
#var_sampling_mask = np.concatenate((var_sampling_mask[:,:,:,np.newaxis],var_sampling_mask[:,:,:,np.newaxis]),\
                                       #   axis = -1)

# White pixels are retrospectively discarded
plt.figure(dpi = 250)
for ii in range(5):
    plt.subplot(1,5,ii+1)
    plt.imshow(var_sampling_mask[ii*10,:,:,0],cmap = "gray")
    plt.axis("off")
plt.show()

print("Undersampling:", 1.0*var_sampling_mask.sum()/var_sampling_mask.size)

#%%
epochs = 100
batch_size= 6

model_name = "C:/Users/DTryfonopoulos/OneDrive - MR Solutions/Documents/PhD/MARKO-PIZURICA_Project/CD-Deep-Cascade-MR-Reconstruction-master/CD-Deep-Cascade-MR-Reconstruction-master/myModels.hdf5"

# Early stopping callback to shut down training after
# 5 epochs with no improvement
earlyStopping = EarlyStopping(monitor='val_loss',
                                       patience=20, 
                                       verbose=0, mode='min')

# Checkpoint callback to save model  along the epochs
checkpoint = ModelCheckpoint(model_name, mode = 'min', \
                         monitor='val_loss',verbose=0,\
                         save_best_only=True, save_weights_only = True)
    
#%% 
# On the fly data augmentation
def combine_generator(gen1,gen2,under_masks):
    while True:
        rec_real = gen1.next()
        rec_imag = gen2.next()
        kspace = np.fft.fft2(rec_real[:,:,:,0]+1j*rec_imag[:,:,:,0])
        kspace2 = np.zeros((kspace.shape[0],kspace.shape[1],kspace.shape[2],2))
        kspace2[:,:,:,0] = kspace.real
        kspace2[:,:,:,1] = kspace.imag
        indexes = np.random.choice(np.arange(under_masks.shape[0], dtype=int), rec_real.shape[0], replace=False)
        kspace2[under_masks[indexes]] = 0
        
        rec_complex = np.zeros((rec_real.shape[0],rec_real.shape[1],rec_real.shape[2],2),dtype = np.float32)
        rec_complex[:,:,:,0] = rec_real[:,:,:,0]
        rec_complex[:,:,:,1] = rec_imag[:,:,:,0]
        
        yield([kspace2,under_masks[indexes].astype(np.float32)],[rec_complex])
        
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


image_generator1 = image_datagen1.flow(rec_train[:,:,:,0,np.newaxis],batch_size = batch_size,seed = seed)
image_generator2 = image_datagen2.flow(rec_train[:,:,:,1,np.newaxis],batch_size = batch_size,seed = seed)        


combined = combine_generator(image_generator1,image_generator2, var_sampling_mask)

# Dispaly sample data augmentation
counter = 0
for ii in combined:
    plt.figure()
    plt.subplot(131)
    plt.imshow(np.abs(ii[1][0][4,:,:,0]+1j*ii[1][0][4,:,:,0]),cmap = 'gray')
    plt.axis("off")
    plt.subplot(132)
    plt.imshow(np.log(1+np.abs(ii[0][0][4,:,:,0] + 1j*ii[0][0][4,:,:,1])),cmap = 'gray')
    plt.axis("off")
    plt.subplot(133)
    plt.imshow(ii[0][0][4,:,:,0].astype(bool),cmap = 'gray')
    plt.axis("off")
    plt.show()
    counter+=1
    if counter > 3:
        break
    
#%%  Undersampling the validation set

indexes = np.random.choice(np.arange(var_sampling_mask.shape[0],dtype =int),kspace_val.shape[0],replace = True)
val_var_sampling_mask = (var_sampling_mask[indexes])
kspace_val[val_var_sampling_mask] = 0

#%%  Training our model

#model = fsnet.deep_cascade_flat_unrolled("ikikii", H, W)
model =fsnet.deep_cascade_flat_unrolled(depth_str = 'ikikii', H=256,W=256,depth = 1,kshape = (3,3), nf = 28,channels = 2)

opt = Adam(lr = 1e-3,decay = 1e-4)
model.compile(loss = 'mse',optimizer=opt)
print(model.summary())

hist = model.fit_generator(combined,
             epochs=epochs,
             steps_per_epoch=rec_train.shape[0]//batch_size,
             verbose=1,
             validation_data= ([kspace_val,val_var_sampling_mask],[rec_val]),
             callbacks=[checkpoint,earlyStopping])