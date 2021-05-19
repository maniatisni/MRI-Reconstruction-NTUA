import matplotlib.pyplot as plt
import numpy as np
import torch
import os 
import glob 
import torchkbnufft as tkbn
import random

dtype = torch.complex64

def DataLoader(data_path, H=256, W=256):
    '''Load All the kSpace files from a given Directory
    and returns the Images in shape
    [#filesm 256,256,2]
     !!!!! data_path ENDING with '/' !!!!!
     '''
    path_all = data_path + '*.npy'
    
    kspace_files = np.asarray(glob.glob(path_all))
    print(kspace_files.shape)
    
    # Shuffle the Data 
    indexes = np.arange(kspace_files.size,dtype = int)
    np.random.shuffle(indexes)
    kspace_files= kspace_files[indexes]
    print(len(kspace_files))    
    
    nfiles = 0
    for ii in range(len(kspace_files)):
        nfiles += np.load(kspace_files[ii]).shape[0] 
        
    images = np.zeros((nfiles,H,W,2))
    aux_counter =0
    norm = np.sqrt(H*W)

    for ii in range(len(kspace_files)):
        aux_kspace = np.load(kspace_files[ii])/norm
        aux = int(aux_kspace.shape[0])
        aux2 = np.fft.ifft2(aux_kspace[:,:,:,0]+ 1j*aux_kspace[:,:,:,1])
        
        
        images[aux_counter:aux_counter+aux,:,:,0] = aux2.real
        images[aux_counter:aux_counter+aux,:,:,1] = aux2.imag
        aux_counter+=aux

    # Shuffle training    
    indexes = np.arange(images.shape[0],dtype = int)
    np.random.shuffle(indexes)
    images = images[indexes]
    print("Number of training samples", images.shape[0], 'Files Shape:', images.shape)
    
    return images

def CreateNuffts(nsamples=256,nspokes=256,dtype = torch.complex64):
    im_size=(nsamples,nspokes)
    grid_size=(nsamples,nspokes)

    nufft_ob = tkbn.KbNufft( im_size=im_size, grid_size=grid_size,).to(dtype) # Creates Radial - kSpace
    adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size,grid_size=grid_size,).to(dtype) # Reconstructs Radial kSpace to IMG 

    return nufft_ob, adjnufft_ob

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
    #plt.plot(kx[:40, :].transpose(), ky[:40, :].transpose())
    #plt.axis('equal')
    #plt.title('k-space trajectory (first 40 spokes)')
    #plt.show()
    
    return ktraj, grid_size, nspokes, spokelength 

def CalcRad_kSpace(image,ktraj,dtype=torch.complex64):
    '''tFkBnufft - Calculates the Radial Like kSpace data'''
    H, W = image.shape[-1], image.shape[-2]
    dim = H,W
    im_size = (H, W)
    grid_size = (H,W) # Can be double the Image size 
    nufft_object = tkbn.KbNufft(im_size=im_size, grid_size=grid_size).to(dtype) # Creates Radial - kSpace
    
    #Convert to Tensor 
    image = torch.tensor(image).to(dtype)#.unsqueeze(0).unsqueeze(0)
    
    kdata = nufft_object(image, ktraj)
    
    return kdata # [1,1,65536]

def CalcRad_kSpace_All(imgdata, traj):
    #H, W = imgdata.shape[-1], imgdata.shape[-2]
        
    kspace_R=np.zeros((imgdata.shape[0],1,1,imgdata.shape[-1]*imgdata.shape[-2]), dtype=np.complex64)
    
    #kspace_R=np.zeros((imgdata.shape[0],imgdata.shape[-1],imgdata.shape[-2],2 ), dtype=np.float32)
    
    # kB_data = np.zeros((len(imgdata),1,1, H*W))
    
    
    for i in range(0,len(imgdata)):
        # Take the Image (1,1,256,256)
        image = imgdata[i,:,:,:,:]
                
        # Calculate Radial kSpace data
        kspace_Rad = np.array(CalcRad_kSpace(image,traj)) #[1,1,65536]
        kspace_R[i,:,:,:] = kspace_Rad
        
        #kspace_Rad2 = np.squeeze(np.reshape(kspace_Rad, (1,1,int(np.sqrt(kspace_Rad.shape[-1])),int(np.sqrt(kspace_Rad.shape[-1]))))) # [256,256]
        #kspace_R[i,:,:,0] = np.real(kspace_Rad2)
        #kspace_R[i,:,:,1] = np.imag(kspace_Rad2)
    
      
    return kspace_R # [Batch, 1,1, 65536]

def Rad2DUndMask(kdata,nsamples = 256,nspokes = 256 ,und=0.8):
        
    kmask = np.ones_like(kdata, dtype=np.int64)
    
    while np.sum(kmask==1)/kdata.shape[-1] > und:
        index = random.randrange(0, nspokes*nsamples-nsamples, nsamples )
        kmask[0,0,index:index+nsamples]= 0
            
    
    degOfUnd = np.sum(kmask==1)/kdata.shape[-1]
    #print('Amount of Sampled Data:', degOfUnd)
    
    return kmask, degOfUnd 


def AllMasks(data):
    sampling_mask_All = np.zeros_like(data, dtype=np.int)
    
    for i in range(0,data.shape[0]):
        kldata = data[i,:,:,:]
        
        mask , defOfUnd = Rad2DUndMask(kldata,und=0.8)
        sampling_mask_All[i,:,:,:] = mask
        
    print('sampling_mask_All:', sampling_mask_All.shape)
    
    return sampling_mask_All ## [Batch, 1,1, 65536]
    

def rstKspace_R(args):
    kspaceR=np.squeeze(args[0])
    mask =np.squeeze( args[1])
    kspaceR[mask] = 0
    out_R = kspaceR
    return out_R


def kSpaceR_US(kSpaceR, Masks, res=False):
    lst_R = []
    
    kSpace_Us=[]
    for i in range(0, kSpaceR.shape[0]):
        
        lst_R.append([kSpaceR[i,:,:,:],Masks[i,:,:,:]])
        kSpace_Us.append(rstKspace_R(lst_R[i]))
    
    kSpace_Us = np.asarray(kSpace_Us)
        
    #Convert to array from list 
    if res:
        kSpace_Us = np.reshape(kSpace_Us,(kSpace_Us.shape[0], int(np.sqrt(kSpace_Us.shape[1])), int(np.sqrt(kSpace_Us.shape[1]))))
    
    return kSpace_Us



def RecImg(us_kdata, traj, dtype=torch.complex64):
    H, W = (int(np.sqrt(us_kdata.shape[-1])), int(np.sqrt(us_kdata.shape[-1]))) 

    im_size = (H, W)
    grid_size = (H,W) # Can be double the Image size 
    adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size,grid_size=grid_size,).to(dtype) # Reconstructs Radial kSpace to IMG 
    
    kdata = np.zeros((1,1,len(us_kdata)), dtype=np.complex64)
    kdata[0,0,:] = us_kdata
    
    
    im_size = (int(np.sqrt(kdata.shape[-1])),int(np.sqrt(kdata.shape[-1])))
    grid_size = im_size
    
    kdata = torch.tensor(kdata).to(torch.complex64)
    dcomp = tkbn.calc_density_compensation_function(ktraj=traj, im_size=im_size)
    #print(dcomp.shape ) # torch [1,1,256,256]
    image_sharp = adjnufft_ob(kdata * dcomp, traj)

    
    return image_sharp
    
    
def RecImgAll(us_kdata,traj):
    images_us = np.zeros((us_kdata.shape[0],int(np.sqrt(us_kdata.shape[-1])),int(np.sqrt(us_kdata.shape[-1]))), dtype=np.complex64)
    
    for i in range (us_kdata.shape[0]):
        kspace = us_kdata[i,:]
        images_us[i,:,:]=np.array(np.squeeze(RecImg(kspace,traj)))
        if i%100 == 0:
            print('done with {i} slices'.format(i))
    return images_us

def ReshComplex_Img(inpData):
    outData = np.zeros((inpData.shape[0],inpData.shape[1],inpData.shape[2],2))
    for i in range(inpData.shape[0]):
       dat_r = np.real(inpData[i,:,:])
       dat_i = np.imag(inpData[i,:,:])
       outData[i,:,:,0]=dat_r
       outData[i,:,:,1]=dat_i

    return outData 

# From (#,1, 1, 65536) to (#,256,256,2)
def ReshComplex_kSp(inpData):
    outData = np.zeros((inpData.shape[0],int(np.sqrt(inpData.shape[-1])),int(np.sqrt(inpData.shape[-1])),2))

    inpData = np.reshape(inpData,(inpData.shape[0],256,256))

    for i in range(inpData.shape[0]):
        dat_r = np.real(inpData[i,:,:])
        dat_i = np.imag(inpData[i,:,:])
        outData[i,:,:,0]=dat_r
        outData[i,:,:,1]=dat_i

    return outData