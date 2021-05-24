#%%
###############################################################
##                 NEEDED DATA DIMENSIONS                    ##
###############################################################

'''  
image: [1,1,256,256] Complex Tensor - dtype = torch.complex64
ktraj: [2, 65536] Float Tensor  - dtype - torch.float32
kspace: [1,1,65536] Complex Tensor  - dtype = torch.complex64  

nufft_ob --> Returns 'Radial kSpace'
adjnufft_ob -->  Returns 'Image'
'''

###############################################################
##                 FUNCTIONS TO LOAD THE FILES               ##
###############################################################
'''
1. DataLoader        - Load the Files                                     - (#, 256, 256, 2)
2. CreateNuffts      - Create the nufft objects (forward & adjoint)       -
3. ImgCreate         - Bring Image in the right format for Nufft          - (#, 1, 1, 256, 256)
4. Rad2DTraj         - Create the Radial Trajectory                       - (2, 65536)
5. CalcRad_kSpace                                                         -
   CalcRad_kSpaceAll - Create the Radial kSpace Data                      - (#,1, 1, 65536)
6. Rad2DUndMask                                                           -
   AllMasks          - Create All undersampling masks                     - (#,1, 1, 65536)
7. rstKspace_R                                                            -
   kSpaceR_US        - Undersample the Radial kSpace Data                 - (#,1, 1, 65536)
8. RecImg                                                                 -
   RecImgAll         - Reconstructs the undersampled kSpace to IMG        - (#,256,256)
'''
###############################################################
##                                                           ##
###############################################################
# %% 
import matplotlib.pyplot as plt
import numpy as np
import torch
import os 
import glob 
import torchkbnufft as tkbn
import random

dtype = torch.complex64

#%% LOAD ALL FILES  FUNCTION

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


#Cartesian
#ImagesTrain = DataLoader('C:/Users/mrsAdmin/Desktop/DATA_Calgary/TRAIN_ALL/')
# ImagesVal = DataLoader('/home/nick/Biodata/DATA/VAL/')

# For Testing - A Small Dataset 
# Images = ImagesVal[0:2,:,:,:]


#%% Create Nufft  objects 

def CreateNuffts(nsamples=256,nspokes=256,dtype = torch.complex64):
    im_size=(nsamples,nspokes)
    grid_size=(nsamples,nspokes)
    
    nufft_ob = tkbn.KbNufft( im_size=im_size, grid_size=grid_size,).to(dtype) # Creates Radial - kSpace
    adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size,grid_size=grid_size,).to(dtype) # Reconstructs Radial kSpace to IMG 
    
    return nufft_ob, adjnufft_ob
    
# nufft_ob, adjnufft_ob =CreateNuffts()
#%% Converts Images in The Format of torchKbNufft

def ImgCreate(data):
    ''' Combines Real&Imag to Image
    (340, 256, 256, 2) -> (340, 1, 1, 256, 256)'''
    data2= np.zeros((np.size(data,0),1,1,np.size(data,1),np.size(data,2)), dtype=np.complex64)
    for i in range(0,len(data)):
        data2[i,0,0,:,:] = data[i,:,:,0] + 1j*data[i,:,:,1]
    return data2 

# img_PyTest = ImgCreate(Images)
# print(img_PyTest.shape)
#%% Create the Radial Trajectory 

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


# ktraj, grid_size, nspokes, spokelength  =  Rad2DTraj()
# # convert k-space trajectory to a tensor
# ktraj = torch.tensor(ktraj).to(torch.float)
# print('ktraj shape: {}'.format(ktraj.shape))   #[2, 65536]

#%%  Creates the RADIAL kSpace Data ktraj (for Each file)
def CalcRad_kSpace(image,ktraj):
    '''tFkBnufft - Calculates the Radial Like kSpace data'''
    # H, W = image.shape[-1], image.shape[-2]
    # dim = H,W
    # im_size = (H, W)
    # grid_size = (H,W) # Can be double the Image size 
    # nufft_ob = tkbn.KbNufft( im_size=im_size, grid_size=grid_size,).to(dtype) # Creates Radial - kSpace
    
    #Convert to Tensor 
    image = torch.tensor(image).to(dtype)#.unsqueeze(0).unsqueeze(0)
    
    kdata = nufft_ob(image, ktraj)
    
    return kdata # [1,1,65536]

#kspDataTest = np.array(CalcRad_kSpace(img_PyTest[0,:,:,:,:],ktraj))
#kspDataTestRes = np.squeeze(np.reshape(kspDataTest, (1,1,256,256)))

# Creates the RADIAL kSpace Data ktraj (for All Files)
def CalcRad_kSpace_All(imgdata, traj,dtype=np.complex64):
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

# kdata_Test = CalcRad_kSpace_All(img_PyTest, ktraj)
#%% Create Radial Undersampling Mask  (for 1 sample)

def Rad2DUndMask(kdata,nsamples = 256,nspokes = 256 ,und=0.8):
        
    kmask = np.ones_like(kdata, dtype=np.int64)
    
    while np.sum(kmask==1)/kdata.shape[-1] > und:
        index = random.randrange(0, nspokes*nsamples-nsamples, nsamples )
        kmask[0,0,index:index+nsamples]= 0
            
    
    degOfUnd = np.sum(kmask==1)/kdata.shape[-1]
    #print('Amount of Sampled Data:', degOfUnd)
    
    return kmask, degOfUnd 

#mask,degOfUnd = Rad2DUndMask(kdata_Test[0,:,:,:],nsamples = 256,nspokes = 256 ,und=0.8)
# Create Radial Undersampling Mask  (for all samples)

def AllMasks(data):
    sampling_mask_All = np.zeros_like(data, dtype=np.int)
    
    for i in range(0,data.shape[0]):
        kldata = data[i,:,:,:]
        
        mask , defOfUnd = Rad2DUndMask(kldata,und=0.8)
        sampling_mask_All[i,:,:,:] = mask
        
    print('sampling_mask_All:', sampling_mask_All.shape)
    
    return sampling_mask_All ## [Batch, 1,1, 65536]
    
# allMasks = AllMasks(kdata_Test)
# plt.imshow(np.squeeze(np.reshape(sampling_mask_All[30,:,:,0], (1,1,256,256))))
#%% Implement the Masking and return Undersampled Data 

def rstKspace_R(args):
    kspaceR=np.squeeze(args[0])
    mask =np.squeeze( args[1])
    kspaceR[mask] = 0
    out_R = kspaceR
    return out_R

# Mask kSpace 

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

# lala = kSpaceR_US(kdata_Test, allMasks, res =False)
#%% Reconstruct Undersampled Radial kSpace to  Image
def RecImg(us_kdata, traj):
    kdata = np.zeros((1,1,len(us_kdata)), dtype=np.complex64)
    kdata[0,0,:] = us_kdata
    
    
    im_size = (int(np.sqrt(kdata.shape[-1])),int(np.sqrt(kdata.shape[-1])))
    grid_size = im_size
    
    kdata = torch.tensor(kdata).to(torch.complex64)
    dcomp = tkbn.calc_density_compensation_function(ktraj=traj, im_size=im_size)
    #print(dcomp.shape ) # torch [1,1,256,256]
    image_sharp = adjnufft_ob(kdata * dcomp, ktraj)

    
    return image_sharp
    
    
def RecImgAll(us_kdata,traj):
    images_us = np.zeros((us_kdata.shape[0],int(np.sqrt(us_kdata.shape[-1])),int(np.sqrt(us_kdata.shape[-1]))), dtype=np.complex64)
    
    for i in range (us_kdata.shape[0]):
        kspace = us_kdata[i,:]
        images_us[i,:,:]=np.array(np.squeeze(RecImg(kspace,ktraj)))
    
    return images_us

# kala = RecImg(lala[1,:],ktraj)
# all_images_final = RecImgAll(lala,ktraj)

# plt.subplot(122)
# plt.imshow(np.abs(all_images_final[1,:,:]), cmap='gray'); plt.title('20% Undersampled')
# plt.subplot(121)
# plt.imshow(np.abs(ImagesVal[1,:,:,0] + 1j* ImagesVal[1,:,:,1]),cmap='gray'); plt.title('Ground Truth')
    


###############################################################
##                 ALL CONVERTIONS                           ##
###############################################################
#%% All Conversions 
import time
start_cell = time.time()
DATA_PATH = '/home/nick/Biodata/DATA/'
# Load the Data (Image Format - (#,256,256,2))
ImagesTrain = DataLoader(DATA_PATH + 'TRAIN/')
ImagesVal = DataLoader(DATA_PATH + 'VAL/')
# ImagesTest = DataLoader(DATA_PATH + 'TEST/')

##################################################################
# Create the Nufft Objects
nufft_ob, adjnufft_ob =CreateNuffts()

##################################################################
# Bring Image in TorchObject Format (#,1,1,256,256)
imgs_PyFormat_Val= ImgCreate(ImagesVal)
print( 'imgs_PyFormat_Val shape', imgs_PyFormat_Val.shape)

imgs_PyFormat_Train= ImgCreate(ImagesTrain)
print( 'imgs_PyFormat_Train shape', imgs_PyFormat_Train.shape)

##################################################################
# Create Radial Trajectory 
ktraj, grid_size, nspokes, nsamples  =  Rad2DTraj()

ktraj = torch.tensor(ktraj).to(torch.float)
print(ktraj.shape)


##################################################################
# Create the Radial k-Space Data (#,1,1,65536)
kdata_Val = CalcRad_kSpace_All(imgs_PyFormat_Val, ktraj)
print('kdata_Val shape:',kdata_Val.shape)

# %%
kdata_Train = CalcRad_kSpace_All(imgs_PyFormat_Train, ktraj)
print('kdata_Train shape',kdata_Train.shape)

##################################################################
# Create Radial Undersampling Mask (#1,1,65536)
Val_Masks = AllMasks(kdata_Val)
print('Val_Masks shape:', Val_Masks.shape)

Train_Masks = AllMasks(kdata_Train)
print('Train_Masks shape:', Train_Masks.shape)

##################################################################
# Create Undersmpled Radial kSpace Data ((#,65536) or(#,256,256))
us_kdata_Val = kSpaceR_US(kdata_Val, Val_Masks, res =False)
print('us_kdata_Val shape:', us_kdata_Val.shape)

us_kdata_Train = kSpaceR_US(kdata_Train, Train_Masks, res =False)
print('us_kdata_Train shape:', us_kdata_Train.shape)

##################################################################
# Create Undersampled Reconstructed Images ((#,256,256))
start = time.time()
us_Img_Val = RecImgAll(us_kdata_Val,ktraj)
print('us_Img_Val shape:', us_Img_Val.shape)
val_time = time.time()-start 

us_Img_Train = RecImgAll(us_kdata_Train,ktraj)
print('us_Img_Train shape:', us_Img_Train.shape)
train_time = time.time()- val_time
end_cell = time.time()
print('Total time:',end_cell-start_cell)
# ImagesTrain, ImagesVal
# kdata_Val, kdata_Train
# us_Img_Train, us_Img_Val
# us_kdata_Val, us_kdata_Train
# %%
