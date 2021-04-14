# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 17:27:54 2021

@author: WScott
"""

t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
tf.shape(t).numpy()



def kSp_AdjNufft(data):
    '''Brings kSpace_Radial Data in the needed format for tkBnufft
    Bring kSpace_val_Rad [340,256,256,2] -> [340,1,1,65536]'''
    
    real = Lambda(lambda data : data[:,:,:,0])(data)
    imag = Lambda(lambda data : data[:,:,:,1])(data)
    kspace_complex = tf.complex(real,imag) # Make complex-valued tensor
    print(kspace_complex.shape)
    kSpR = tf.reshape(kspace_complex, [kspace_complex.shape[0],1,1, kspace_complex.shape[1]*kspace_complex.shape[2]])
    return kSpR
    
aka = kSp_AdjNufft(kspace_val_Rad2)

data = kspace_val_Rad2
real = Lambda(lambda data : data[:,:,:,0])(data)
imag = Lambda(lambda data : data[:,:,:,1])(data)
kspace_complex = tf.complex(real,imag) # Make complex-valued tensor
print(kspace_complex.shape)
kSpR = tf.reshape(kspace_complex, [kspace_complex.shape[0],1,1, kspace_complex.shape[1]*kspace_complex.shape[2]])



def kSp_AdjNufft2(data):
    '''Brings kSpace_Radial Data in the needed format for tkBnufft
    Bring kSpace_val_Rad [340,256,256,2] -> [340,1,1,65536]'''
    
    
    d1 = data[:,:,:,0]
    d2 = data[:,:,:,1]
    dd = d1 + 1j*d2
    ddK = np.reshape(dd, (dd.shape[0],dd.shape[1]*dd.shape[2]))
    kSpR = np.zeros((ddK.shape[0],1,1,ddK.shape[1]), dtype=np.complex64)
    kSpR[:,0,0,:] = ddK
    return kSpR


def kSp_AdjNufft(data):
    '''Brings kSpace_Radial Data in the needed format for tkBnufft
    Bring kSpace_val_Rad [340,256,256,2] -> [340,1,1,65536]'''
    
    real = Lambda(lambda data : data[:,:,:,0])(data)
    imag = Lambda(lambda data : data[:,:,:,1])(data)
    kspace_complex = tf.complex(real,imag) # Make complex-valued tensor
    #kSpR = tf.reshape(kspace_complex, [kspace_complex.numpy().shape[0],1,1, kspace_complex.numpy().shape[1]*kspace_complex.numpy().shape[2]])
    kSpR = tf.reshape(kspace_complex, [kspace_complex.shape[0],1,1, kspace_complex.shape[1]*kspace_complex.shape[2]])
    return kSpR

# Adjnufft Back
def AdjNufftBack2(data, ktraj, method=3):
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

def AdjNufftBack(data, ktraj, method=3):
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
