import tensorflow as tf
import os 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Lambda, Add, LeakyReLU,  \
                                    MaxPooling2D, concatenate, UpSampling2D,\
                                    Multiply, ZeroPadding2D, Cropping2D, Reshape

# Use for Radial
from Modules.All_Radial_Functions import Rad2DTraj, AdjNufftBack
#from Modules.All_Radial_Functions import *

#os.chdir('D:/DIMITRIS_2021/tfkbnufft_master')
from tfkbnufft_master.tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft_master.tfkbnufft.kbnufft import KbNufftModule
from tfkbnufft_master.tfkbnufft.mri.dcomp_calc import calculate_radial_dcomp_tf, calculate_density_compensator
import tfkbnufft



def fft_layer(image):
    # get real and imaginary portions
    real = Lambda(lambda image: image[:, :, :, 0])(image)
    imag = Lambda(lambda image: image[:, :, :, 1])(image)

    image_complex = tf.complex(real, imag)  # Make complex-valued tensor
    kspace_complex = tf.signal.fft2d(image_complex)

    # expand channels to tensorflow/keras format
    real = tf.expand_dims(tf.math.real(kspace_complex), -1)
    imag = tf.expand_dims(tf.math.imag(kspace_complex), -1)
    kspace = tf.concat([real, imag], -1)
    return kspace




def ifft_layer(kspace_2channel):
    
    #get real and imaginary portions
    real = Lambda(lambda kspace_2channel : kspace_2channel[:,:,:,0])(kspace_2channel)
    imag = Lambda(lambda kspace_2channel : kspace_2channel[:,:,:,1])(kspace_2channel)
    
    kspace_complex = tf.complex(real,imag) # Make complex-valued tensor [None,256,256]
    image_complex = tf.signal.ifft2d(kspace_complex) #tensor[none,256,256]
    
    
    # expand channels to tensorflow/keras format
    real = tf.expand_dims(tf.math.real(image_complex),-1) # tensor [None, 256, 256, 1]
    imag = tf.expand_dims(tf.math.imag(image_complex),-1)
    # generate 2-channel representation of image domain
    image_complex_2channel = tf.concat([real, imag], -1)
    
    return image_complex_2channel # tensor [None,256,256,2]

# I ADDED this PART 
def ifft_layer_R(kspace_2channel):
    real = Lambda(lambda kspace_2channel : kspace_2channel[:,:,:,0])(kspace_2channel)
    imag = Lambda(lambda kspace_2channel : kspace_2channel[:,:,:,1])(kspace_2channel)
    
    kspace_complex = tf.complex(real,imag) # Make complex-valued tensor [None,256,256]
    
    dim = kspace_2channel.shape[1],kspace_2channel.shape[2]
    nufft_ob = KbNufftModule(im_size=dim, grid_size=dim, norm='ortho')
    
    ktraj, grid_size, nspokes, spokelength = Rad2DTraj(nsamples=256, nspokes=256, ga=True)
    #ktraj = tf.convert_to_tensor(ktraj, dtype = tf.float32)[None, ...]
    ktraj = tf.constant(ktraj, dtype = tf.float32)[None, ...]
    
    
    kspaceComplex = Reshape((-1,1,1,256*256))(kspace_complex) # ? ? kdata[1,1,256x256]
    adjoint_op = kbnufft_adjoint(nufft_ob._extract_nufft_interpob())
    
    
    #dim = data.shape[1],data.shape[2]
    #nufft_ob = KbNufftModule(im_size=dim, grid_size=dim, norm='ortho')
    #adjoint_op = kbnufft_adjoint(nufft_ob._extract_nufft_interpob())
    #for _ in range(kspaceComplex.shape[0]):
     #    x = adjoint_op(y, ktraj)
    
    #image_complex_2channel = AdjNufftBack(kspace_2channel, ktraj, method=3)
    #image_complex_2channel = image_complex_2channel.numpy()
    #image_complex_2channel = tf.convert_to_tensor(image_complex_2channel)
    return image_complex_2channel

#Dims for nufft:  ktraj[1,2,256x256] / kdata[1,1,256x256]/ image [1,1,256,256]
def AdjNufftBack(data, ktraj, method=3):
    ''' Receives the kSpace_Radial Data and returns the Reconstructed Image'''
    start = time.time()
    dim = data.shape[1],data.shape[2]
    nufft_ob = KbNufftModule(im_size=dim, grid_size=dim, norm='ortho')
    
    interpob = nufft_ob._extract_nufft_interpob()
    nufft_adj = kbnufft_adjoint(interpob)
    user:TRDim nikos maniatis
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


def cnn_block(cnn_input, depth, nf, kshape,channels):
    """
    :param cnn_input: Input layer to CNN block
    :param depth: Depth of CNN. Disregarding the final convolution block that goes back to
    2 channels
    :param nf: Number of filters of convolutional layers, except for the last
    :param kshape: Shape of the convolutional kernel
    :return: 2-channel, complex reconstruction
    """
    layers = [cnn_input]

    for ii in range(depth):
        # Add convolutional block
        layers.append(Conv2D(nf, kshape, padding='same')(layers[-1]))
        layers.append(LeakyReLU(alpha=0.1)(layers[-1]))
    final_conv = Conv2D(channels, (1, 1), activation='linear')(layers[-1])
    rec1 = Add()([final_conv,cnn_input])
    return rec1


def unet_block(unet_input, kshape=(3, 3),channels = 2):
    """
    :param unet_input: Input layer
    :param kshape: Kernel size
    :return: 2-channel, complex reconstruction
    """

    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(unet_input)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(up1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(up2)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(up3)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)

    conv8 = Conv2D(channels, (1, 1), activation='linear')(conv7)
    out = Add()([conv8, unet_input])
    return out

def unet_block2(unet_input, kshape=(3, 3),channels = 2):
    """
    :param unet_input: Input layer
    :param kshape: Kernel size
    :return: 2-channel, complex reconstruction
    """

    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(unet_input)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(96, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(96, kshape, activation='relu', padding='same')(conv2)
    conv2 = Conv2D(96, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(192, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(192, kshape, activation='relu', padding='same')(conv3)
    conv3 = Conv2D(192, kshape, activation='relu', padding='same')(conv3)
    
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv2D(96, kshape, activation='relu', padding='same')(up1)
    conv4 = Conv2D(96, kshape, activation='relu', padding='same')(conv4)
    conv4 = Conv2D(96, kshape, activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv2D(48, kshape, activation='relu', padding='same')(up2)
    conv5 = Conv2D(48, kshape, activation='relu', padding='same')(conv5)
    conv5 = Conv2D(48, kshape, activation='relu', padding='same')(conv5)

    conv6 = Conv2D(channels, (1, 1), activation='linear')(conv5)
    out = Add()([conv6, unet_input])
    return out

def DC_block(rec,mask,sampled_kspace,channels,kspace = False):
    """
    :param rec: Reconstructed data, can be k-space or image domain
    :param mask: undersampling mask
    :param sampled_kspace:
    :param kspace: Boolean, if true, the input is k-space, if false it is image domain
    :return: k-space after data consistency
    """

    if kspace:
        rec_kspace = rec
    else:
        rec_kspace = Lambda(fft_layer)(rec)
    rec_kspace_dc =  Multiply()([rec_kspace,mask])
    rec_kspace_dc = Add()([rec_kspace_dc,sampled_kspace])
    return rec_kspace_dc

def deep_cascade_flat_unrolled(depth_str = 'ikikii', H=256,W=256,depth = 5,kshape = (3,3), nf = 48,channels = 2):
    """
    :param depth_str: string that determines the depth of the cascade and the domain of each
    subnetwork
    :param H: Image heigh
    :param W: Image width
    :param kshape: Kernel size
    :param nf: number of filters in each convolutional layer
    :return: Deep Cascade Flat Unrolled model
    """

    inputs = Input(shape=(H,W,channels))
    mask = Input(shape=(H,W,channels))
    layers = [inputs]
    kspace_flag = True
    for ii in depth_str:
        
        if ii =='i':
            # Add IFFT
            layers.append(Lambda(ifft_layer)(layers[-1]))
            kspace_flag = False
        # Add CNN block
        layers.append(cnn_block(layers[-1],depth,nf,kshape,channels))

        # Add DC block
        layers.append(DC_block(layers[-1],mask,inputs,channels,kspace=kspace_flag))
        kspace_flag = True
    out = Lambda(ifft_layer)(layers[-1])
    model = Model(inputs=[inputs,mask], outputs=out)
    return model

# I added this folder  for Radial RECON 
def deep_cascade_flat_unrolled_R(depth_str = 'ikikii', H=256,W=256,depth = 5,kshape = (3,3), nf = 48,channels = 2):
    """
    :param depth_str: string that determines the depth of the cascade and the domain of each
    subnetwork
    :param H: Image heigh
    :param W: Image width
    :param kshape: Kernel size
    :param nf: number of filters in each convolutional layer
    :return: Deep Cascade Flat Unrolled model
    """

    inputs = Input(shape=(H,W,channels))
    mask = Input(shape=(H,W,channels))
    layers = [inputs]
    kspace_flag = True
    for ii in depth_str:
        
        if ii =='i':
            # Add IFFT
            layers.append(Lambda(ifft_layer_R)(layers[-1]))
            kspace_flag = False
        # Add CNN block
        layers.append(cnn_block(layers[-1],depth,nf,kshape,channels))

        # Add DC block
        layers.append(DC_block(layers[-1],mask,inputs,channels,kspace=kspace_flag))
        kspace_flag = True
    out = Lambda(ifft_layer_R)(layers[-1])
    model = Model(inputs=[inputs,mask], outputs=out)
    return model

def deep_cascade_unet(depth_str='ki', H=218, W=170, Hpad = 3, Wpad = 3, kshape=(3, 3),channels = 22):

    inputs = Input(shape=(H,W,channels))
    mask = Input(shape=(H,W,channels))
    layers = [inputs]
    kspace_flag = True
    for ii in depth_str:
        
        if ii =='i':
            # Add IFFT
            layers.append(Lambda(ifft_layer)(layers[-1]))
            kspace_flag = False
        # Add CNN block
        layers.append(ZeroPadding2D(padding=(Hpad,Wpad))(layers[-1]))
        layers.append(unet_block(layers[-1], kshape,channels))
        layers.append(Cropping2D(cropping=(Hpad,Wpad))(layers[-1]))
        
        # Add DC block
        layers.append(DC_block(layers[-1],mask,inputs,channels,kspace=kspace_flag))
        kspace_flag = True
    out = Lambda(ifft_layer)(layers[-1])
    model = Model(inputs=[inputs,mask], outputs=out)
    return model
