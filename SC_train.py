# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 16:19:40 2021

@author: WScott
"""

# %% [markdown]
# # Flat Unrolled Cascade - Single-channel - Train
# 
# - Single-channel data
# - Images are 256x256
# - R=5

# %% [code]
# print(os.listdir("../"))

# %% [code]
import gc
gc.enable()
gc.collect()

# %% [code]
# TEST FOR GPU 
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %% [code]
tf.__version__


# %% [code]
tf.debugging.set_log_device_placement(True)

# %% [code]
# Go In this Dir to load all the Needed Files 
import os 
os.chdir('../input/cd-deepcascade')

# %% [code]
# Import All py Files (Scripts-Functions)
from os.path import dirname, basename, isfile, join
import glob
modules = glob.glob(join(dirname(''), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

# %% [code]
gc.enable()
del modules, __all__
gc.collect()

# %% [code]
#%matplotlib inline
import matplotlib.pylab as plt
import numpy as np
import os
import glob
import sys
import tensorflow as tf

# Importing our model
MY_UTILS_PATH = "../Modules/"
if not MY_UTILS_PATH in sys.path:
    sys.path.append(MY_UTILS_PATH)
    
import Modules.cs_models_sc as fsnet

# Importing callbacks and data augmentation utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# %% [code]
os.chdir('/kaggle/input')

# %% [code]
## PARAMETERS
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

# %% [code]
# Keep only k_space_files_val, kspace_files_train
del val_path, indexes
gc.collect()

# %% [code]


# %% [code]
# Get number of samples
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

# %% [code]
# Keep Only rec_train 
del  aux_kspace, aux, aux2 , aux_counter, indexes, kspace_files_train
gc.collect()

# %% [code]
# Get number of samples
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

print("Number of val samples", kspace_val.shape[0])

# %% [code]
del nval, aux_kspace, aux, aux2,  aux_counter, kspace_files_val
gc.collect()

# %% [code]
# My Sampling Pattern 
#und = 0.1 
def Poisson(dim, und=0.8):
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
    #und_mask =tf.convert_to_tensor(C)
    return und_mask

# %% [code]
C = Poisson(256)
var_sampling_mask = (np.fft.fftshift(C, axes=(1,2)))

# %% [code]
del C
gc.collect()

# %% [code]
# Loading sampling patterns. Notice that here we are using uncentred k-space

#var_sampling_mask = np.fft.fftshift(~np.load("../input/calgarycampinas-poissonsampling-implemented/R10_218x174.npy") \
      #                              ,axes = (1,2))

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
import h5py
import numpy as np
f = h5py.File("mytestfile1.hdf5", "w")

# %% [code]
# LOAD the file to SAVE the TRAINED MODEL 
epochs = 100
batch_size= 6

#model_name = "../Models/flat_unrolled_cascade_ikikii.hdf5"
#model_name = '../input/cd-deepcascade-sc-savemodel/myModels.hdf5'
#model_name ='D:/DIMITRIS_2021/mytestfile.hdf5'
model_name = ' D:/DIMITRIS_2021/Modules/file_20210409.hdf5'
# Early stopping callback to shut down training after
# 5 epochs with no improvement
earlyStopping = EarlyStopping(monitor='val_loss',
                                       patience=20, 
                                       verbose=0, mode='min')

# Checkpoint callback to save model  along the epochs
checkpoint = ModelCheckpoint(model_name, mode = 'min', \
                         monitor='val_loss',verbose=0,\
                         save_best_only=True, save_weights_only = True)

# %% [code]
tf.device('/physical_device:GPU:0')

# %% [code]
# On the fly data augmentation
import scipy
tf.debugging.set_log_device_placement(True)

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

#%% Dispaly sample data augmentation
counter = 0 # We need the counter because the ImageDataGenerator loops infinitely 
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

# %% [code]
indexes = np.random.choice(np.arange(var_sampling_mask.shape[0],dtype =int),kspace_val.shape[0],replace = True)
val_var_sampling_mask = (var_sampling_mask[indexes])
val_var_sampling_mask = np.asarray(val_var_sampling_mask, dtype=bool)

# %% [markdown]
# # It is not working in Kaggle 
# import multiprocess as mp
# import tqdm
# 
# 
# p = mp.Pool(mp.cpu_count())
# result = p.map(rstKspace, tqdm(lst))
# p.close()
# p.join()

# %% [code]
# Create a list with kSpace and Sampling Masks 
lst=[]
for i in range(0, len(kspace_val)):
    lst.append([kspace_val[i,:,:,:],val_var_sampling_mask[i,:,:,:]])

# %% [code]
np.shape(lst)

#np.shape(lst)
# lst[1699][0] [:,:,0]

# %% [code]
# Function to make zero all the entries 

def rstKspace(args):
    kspace_val=args[0]
    val_var_sampling_mask = args[1]
    kspace_val[val_var_sampling_mask] = 0
    out = kspace_val
    return out

# %% [code]
# Mask k-Space 

kSpace=[]
for i in range(0, len(kspace_val)):
    kSpace.append(rstKspace(lst[i]))
    
#Convert to array from list 
kspace_val = np.asarray(kSpace)

# %% [code]
del kSpace, lst 
gc.collect()

# %% [code]
# Training our model

model = fsnet.deep_cascade_flat_unrolled("ikikii", H, W)
opt = Adam(lr = 1e-3,decay = 1e-4)
model.compile(loss = 'mse',optimizer=opt)
print(model.summary())


#steps_per_epoch:  value as the total number of training data points divided by the batch size
hist = model.fit_generator(combined,
             epochs=epochs,
             steps_per_epoch=rec_train.shape[0]//batch_size, # Needed for the ImageDataGenerator not to run for ever
             verbose=1,
             validation_data= ([kspace_val,val_var_sampling_mask],[rec_val]),
             callbacks=[checkpoint,earlyStopping])