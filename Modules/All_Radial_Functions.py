# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:04:46 2021

@author: DTryfonopoulos
"""

import matplotlib.pylab as plt
import numpy as np
import os
import glob
import sys
import random 
import time 
import tfkbnufft

#import cs_models_sc as fsnet

# Importing callbacks and data augmentation utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import  Adam
import tensorflow as tf
from tensorflow.keras.layers import Lambda


from tfkbnufft_master.tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft_master.tfkbnufft.kbnufft import KbNufftModule
from tfkbnufft_master.tfkbnufft.mri.dcomp_calc import calculate_radial_dcomp_tf, calculate_density_compensator



def ImgCreate(data):
    ''' Combines Real&Imag to Image
    (340, 256, 256, 2) -> (340, 1, 1, 256, 256)'''
    data2= np.zeros((np.size(data,0),1,1,np.size(data,1),np.size(data,2)), dtype=np.complex64)
    for i in range(0,len(data)):
        data2[i,0,0,:,:] = data[i,:,:,0] + 1j*data[i,:,:,1]
    return data2 


def Rad2DTraj(nsamples=256, nspokes=256, ga=True):
    
    spokelength = nsamples
    grid_size= (spokelength, spokelength)
    
    # nspokes = # 256,300
    # ga = 
    
    if ga:
        ga = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))
    else:
        ga = np.deg2rad(180 / nspokes)
        
    kx = np.zeros(shape=(spokelength, nspokes))
    ky = np.zeros(shape=(spokelength, nspokes))
    ky[:, 0] = np.linspace(-np.pi, np.pi, spokelength)
    for i in range(1, nspokes):
        kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
        ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]
    
    ky = np.transpose(ky)
    kx = np.transpose(kx)

    ktraj = np.stack((ky.flatten(), kx.flatten()), axis=0)

    # Plots
   # plt.plot(kx[:40, :].transpose(), ky[:40, :].transpose())
    #plt.axis('equal')
    #plt.title('k-space trajectory (first 40 spokes)')
    #plt.show()
    
    return ktraj, grid_size, nspokes, spokelength 


def CalcRad_kSpace(image,ktraj):
    '''tFkBnufft - Calculates the Radial Like kSpace data'''
    H, W = image.shape[-1], image.shape[-2]
    dim = H,W
    nufft_ob = KbNufftModule(im_size=dim, grid_size=dim, norm='ortho')
    kdata = kbnufft_forward(nufft_ob._extract_nufft_interpob())(image, ktraj)

    # add some noise (robustness test)
    siglevel = tf.reduce_mean(tf.math.abs(kdata))
    kdata = kdata + tf.cast((siglevel/5) *tf.random.normal(kdata.shape, dtype=siglevel.dtype), kdata.dtype)
    #print('kdata Shape:', kdata.shape)
    return kdata


def ReImCreate(data):
    ''' Rerurns the Radial kSpace in the needed CD_Cascade format '''
    data2 = data.numpy()
    data2 = np.reshape(data2, (1,1,256,256)) # (1,1,nspokes,nsamples)
    
    data3 = np.zeros((256,256,2))
    data3[:,:,0]= np.squeeze(np.real(data2))
    data3[:,:,1]= np.squeeze(np.imag(data2))
    #print('Output Data Shape:', data3.shape)
    return data3
    
# datata = ReImCreate(kdata)
# plt.imshow(np.abs((datata[:,:,0]+ 1j*datata[:,:,1]))) 

def CD_kSpaceRad(imgdata, traj):
    #H, W = imgdata.shape[-1], imgdata.shape[-2]
        
    #kspace_val_Rad=np.zeros_like(imgdata)
    kspace_val_Rad=np.zeros((imgdata.shape[0],imgdata.shape[-1],imgdata.shape[-2],2 ), dtype=np.float32)
    
    # kB_data = np.zeros((len(imgdata),1,1, H*W))
    
    
    for i in range(0,len(imgdata)):
        # Take the Image (1,1,256,256)
        #image = tf.convert_to_tensor(data[i,:,:,:,:])
        image = imgdata[i,:,:,:,:]
        # Calculate Radial kSpace data
        kdata = CalcRad_kSpace(image,traj)
        # kB_data[i,:,:,:] = kdata #Data for reconstructing image (nufft)
        
        # Return the needed format for CD_DeepCascade
        data_Rad = ReImCreate(kdata)
        
        # Store data
        kspace_val_Rad[i,:,:,:] = data_Rad
        #print('Progress: {}/{}'.format(i, len(imgdata)))
        
    return kspace_val_Rad


def Rad2DUndMask(kdata,nsamples,nspokes,und=0.8):
    #kdata2= kdata.numpy()
    kmask = np.ones_like(kdata, dtype=np.int64)
    kmask2 = np.zeros((nspokes,nsamples,2),dtype=np.int64)
    
    while np.sum(kmask==1)/kdata.shape[-1] > und:
        index = random.randrange(0, nspokes*nsamples-nsamples, nsamples )
        kmask[0,0,index:index+nsamples]= 0
            
    kmask2[:,:,0] =np.squeeze(kmask.reshape(1,1,256,256))
    kmask2[:,:,1] =np.squeeze(kmask.reshape(1,1,256,256))
    
    degOfUnd = np.sum(kmask==1)/kdata.shape[-1]
    #print('Amount of Sampled Data:', degOfUnd)
    
    return kmask2, degOfUnd 

# mask , defOfUnd = Rad2DUndMask(kdata,256,256,und=0.7)


def kSp_AdjNufft(data):
    '''Brings kSpace_Radial Data in the needed format for tkBnufft
    Bring kSpace_val_Rad [340,256,256,2] -> [340,1,1,65536]'''
    
    
    d1 = data[:,:,:,0]
    d2 = data[:,:,:,1]
    print('data',data.shape)
    print('d1',d1.shape)
    dd = np.zeros((d1.shape))
    for i in range(d2.shape[0]):
        dd[i,:,:] = d1[i,:,:] + 1j*d2[i,:,:]
    ddK = np.reshape(dd, (dd.shape[0],dd.shape[1]*dd.shape[2]))
    kSpR = np.zeros((ddK.shape[0],1,1,ddK.shape[1]), dtype=np.complex64)
    kSpR[:,0,0,:] = ddK # [..,1,1,256*256]
    kSpR = tf.convert_to_tensor(kSpR)
    return kSpR


def kSp_AdjNufft2(data):
    '''Brings kSpace_Radial Data in the needed format for tkBnufft
    Bring kSpace_val_Rad [340,256,256,2] -> [340,1,1,65536]'''
    
    real = Lambda(lambda data : data[:,:,:,0])(data)
    imag = Lambda(lambda data : data[:,:,:,1])(data)
    kspace_complex = tf.complex(real,imag) # Make complex-valued tensor
    print(kspace_complex.shape)
    #kSpR = tf.reshape(kspace_complex, [kspace_complex.numpy().shape[0],1,1, kspace_complex.numpy().shape[1]*kspace_complex.numpy().shape[2]])
    kSpR = tf.reshape(kspace_complex, [kspace_complex.shape[0],1,1, kspace_complex.shape[1]*kspace_complex.shape[2]])
    return kSpR

# Adjnufft Back
def AdjNufftBack(data, ktraj, method=3):
    ''' Receives the kSpace_Radial Data and returns the Reconstructed Image'''
    start = time.time()
    dim = data.shape[1],data.shape[2]
    nufft_ob = KbNufftModule(im_size=dim, grid_size=dim, norm='ortho')
    
    
    interpob = nufft_ob._extract_nufft_interpob()
    nufft_adj = kbnufft_adjoint(interpob)
    
    #image_sharp_all = np.zeros_like((data))
    image_sharp_all = 0* (data)
    data = kSp_AdjNufft(data)
        
    for i in range(0,data.shape[0]):
    #for i in range(0,1):
        #dataA = data[i,:,:,:]
        dataA = data.numpy()[i,:,:,:]
        
    
        if method ==1:
            dc = ' No DC'
            # method 1: No density compensation
            image_sharp = nufft_adj(dataA, ktraj)
        elif method==2:
            dc = ' Simple DC'
            # method 2: use density compensation
            dcomp = calculate_radial_dcomp_tf(interpob, kbnufft_forward(interpob), nufft_adj, ktraj[0])[None, :]
            image_sharp = nufft_adj(dataA * tf.cast(dcomp, dataA.dtype), ktraj)
        elif method==3:
            dc = ' Pipe DC'
            # method 3: Pipe density compensation
            dcomp_new = calculate_density_compensator(interpob, kbnufft_forward(interpob), nufft_adj, ktraj[0])[None, :]
            image_sharp = nufft_adj(dataA * tf.cast(dcomp_new, dataA.dtype), ktraj)
        else:
            dc='---Error---'
            print('Please provide the right Method number')
        
        
        image_sharp_all[i,:,:,0] = np.real(np.squeeze(image_sharp.numpy()))
        image_sharp_all[i,:,:,1] = np.imag(np.squeeze(image_sharp.numpy()))
        print(i, time.time() - start)
        
    
    real = Lambda(lambda image_sharp_all : image_sharp_all[:,:,:,0])(image_sharp_all)
    imag = Lambda(lambda image_sharp_all : image_sharp_all[:,:,:,1])(image_sharp_all)
    
    real = tf.expand_dims(real,-1)
    imag = tf.expand_dims(imag,-1)
    image_complex = tf.concat([real, imag], -1)

    
    return image_complex

def AdjNufftBack2(data, ktraj, method=3):
    ''' Receives the kSpace_Radial Data and returns the Reconstructed Image'''
    #start = time.time()
    dim = data.shape[1],data.shape[2]
    nufft_ob = KbNufftModule(im_size=dim, grid_size=dim, norm='ortho')
    
    
    interpob = nufft_ob._extract_nufft_interpob()
    nufft_adj = kbnufft_adjoint(interpob)
    
    #image_sharp_all = np.zeros_like((data))
    image_sharp_all = 0*data 
    
    data = kSp_AdjNufft(data)
    data = tf.convert_to_tensor(data, dtype=tf.complex64)
        
    for i in range(0,data.shape[0]):
    #for i in range(0,1):
        #dataA = data[i,:,:,:]
        dataA = data.numpy()[i,:,:,:]
        
    
        if method ==1:
            dc = ' No DC'
            # method 1: No density compensation
            image_sharp = nufft_adj(dataA, ktraj)
        elif method==2:
            dc = ' Simple DC'
            # method 2: use density compensation
            dcomp = calculate_radial_dcomp_tf(interpob, kbnufft_forward(interpob), nufft_adj, ktraj[0])[None, :]
            image_sharp = nufft_adj(dataA * tf.cast(dcomp, dataA.dtype), ktraj)
        elif method==3:
            dc = ' Pipe DC'
            # method 3: Pipe density compensation
            dcomp_new = calculate_density_compensator(interpob, kbnufft_forward(interpob), nufft_adj, ktraj[0])[None, :]
            image_sharp = nufft_adj(dataA * tf.cast(dcomp_new, dataA.dtype), ktraj)
        else:
            dc='---Error---'
            print('Please provide the right Method number')
        
        
        image_sharp_all[i,:,:,0] = np.real(np.squeeze(image_sharp.numpy()))
        image_sharp_all[i,:,:,1] = np.imag(np.squeeze(image_sharp.numpy()))
        #print(i, time.time() - start)
        
    image_sharp_all = tf.convert_to_tensor(image_sharp_all, dtype=np.float32)
    
    return image_sharp_all


def ifft_layer_R2(kspace_2channel):
    ktraj, grid_size, nspokes, spokelength = Rad2DTraj(nspokes=256, ga=True)
    ktraj = tf.convert_to_tensor(ktraj)[None, ...]
    image_complex_2channel = AdjNufftBack(kspace_2channel, ktraj, method=3)
    return image_complex_2channel


def plotRadRecon(image, degOfUnd=1 ):
    # show the images
    print('Amount of NonAcquired Data:',degOfUnd )

    image_sharp_numpy = np.squeeze(image)
    
    plt.figure(figsize=(10,20))

    plt.imshow(np.absolute(image_sharp_numpy))
    plt.gray
    plt.title('With DC')

    plt.show()
    
    
