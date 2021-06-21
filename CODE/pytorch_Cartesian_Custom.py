# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 10:22:02 2021

@author: mrsAdmin
"""
# %%
import torch
import os 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt
torch.manual_seed(42)
from tqdm import tqdm
# os.chdir('C:/Users/mrsAdmin/Desktop/DIMITRIS_2021/DL_MRI_QinSchlempler')
from dnn_Cartesian_Custom import  undersampling_rate, create_dummy_data, iterate_minibatch, prep_input, \
                          cartesian_mask, from_tensor_format, complex_psnr, CS_MRI_MRF_REC
                          
# os.chdir('C:/Users/mrsAdmin/Desktop/DIMITRIS_2021')
from DataFunctions import DataLoader

import numpy as np
np.random.seed(42)
import time
os.chdir('/home/nick/Biodata/DATA')
# %% 
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
Tensor = torch.cuda.FloatTensor if use_cuda else torch.Tensor

#Nx, Ny, Nt = 256, 256, 1
Ny_red = 4 # Reduction in Ny Dimension 
acc = 4.0
num_epoch = 30
batch_size = 10
save_fig = True
debug_mode = False
save_every = 2

def pikla(list,name):
    os.chdir('/home/nick/bioFinal/fixedLambda/')
    with open(name+'.pkl', 'wb') as f:
        pickle.dump(list, f)
    os.chdir('/home/nick/Biodata/DATA')


# %%
# Load dataset - Image Format 
VALID = DataLoader('./VAL/')
TRAIN = DataLoader('./TRAIN/')
TEST = DataLoader('./TEST/')

# THIS TOOK 1 MIN/EPOCH
# VALID = VALID[:800,:,:,:]
# TRAIN = TRAIN[:2500,:,:,:]
# TEST = TEST[:400,:,:,:]

VALID = VALID[:600,:,:,:]
TRAIN = TRAIN[:2000,:,:,:]
TEST = TEST[:400,:,:,:]

VALID_comp = VALID[:,:,:,0] + 1j* VALID[:,:,:,1]
TRAIN_comp = TRAIN[:,:,:,0] + 1j* TRAIN[:,:,:,1]
TEST_comp = TEST[:,:,:,0] + 1j* TEST[:,:,:,1]
del VALID
del TRAIN
del TEST

# %%
train, validate, test =TRAIN_comp, VALID_comp, TEST_comp
# train_batch_size = int(np.round(train.shape[0] / 10))
# val_batch_size = int(np.round(validate.shape[0] / 2))
# test_batch_size = int(np.round(test.shape[0] / 2))
train_batch_size = 64
val_batch_size = 64
test_batch_size = 64

Nx, Ny = 256, 256

for acc in [2.0,4.0,8.0]:
    # Test creating mask and compute the acceleration rate
    dummy_mask = cartesian_mask(( Nx, Ny), acc, sample_n=8) #(10, 256, 64)
    sample_und_factor = undersampling_rate(dummy_mask)
    print('Undersampling Rate: {:.2f}'.format(sample_und_factor))  # 0.25 US Rate 

    model = CS_MRI_MRF_REC(ns=2, nf=5).to(device) # Do Not Show DC in this setting because there is nothing to Learn
    criterion = torch.nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(0.001), betas=(0.5, 0.999))
    
    print('Start Training...')
    trainERROR =[]
    validERROR =[]
    testERROR = [] 
    trainPSNR = []
    trainBasePSNR = []
    validPSNR = []
    validBasePSNR = []
    best_valid_loss = float('inf')

    i = 0
    start = time.time()
    # TRAINING 
    for epoch in tqdm(range(num_epoch)):
        epoch_start = time.time()
        t_start = time.time()
        # Training
        train_err = 0
        train_batches = 0
        base_psnr = 0
        train_psnr = 0
            
        for im in iterate_minibatch(train, train_batch_size, shuffle=True):
            im_und, k_und, mask, im_gnd = prep_input(im, acc) # im [2,2,30,256,32] [#,Channels,Nt,Nx,Ny]
            # im_und, k_und, mask, im_gnd -> [2,2,256,32,30]
            im_u = Variable(im_und.type(Tensor))
            k_u = Variable(k_und.type(Tensor))
            mask = Variable(mask.type(Tensor))
            gnd = Variable(im_gnd.type(Tensor))

            optimizer.zero_grad()
            rec = model(im_u, k_u, mask)
            loss = criterion(rec, gnd)
            loss.backward()
            optimizer.step()
            for im_gnd, im_und, rec in zip(im,
                                        from_tensor_format(im_und.numpy()),
                                        from_tensor_format(rec.data.cpu().numpy())):
                    base_psnr += complex_psnr(im_gnd, im_und, peak='max')
                    train_psnr += complex_psnr(im_gnd, rec, peak='max')
                
            train_err += loss.item()
            train_batches += 1

            if debug_mode and train_batches == 20:
                break
        trainPSNR.append(train_psnr/train_batches)
        trainBasePSNR.append(base_psnr/train_batches)
        trainERROR.append(train_err/train_batches)
            
        # VALIDATION
        validate_err = 0
        validate_batches = 0
        valid_psnr = 0
        valid_base_psnr = 0
        model.eval()
        
        for im in iterate_minibatch(validate, val_batch_size, shuffle=False):
            im_und, k_und, mask, im_gnd = prep_input(im, acc)
            with torch.no_grad():
                im_u = Variable(im_und.type(Tensor))
                k_u = Variable(k_und.type(Tensor))
                mask = Variable(mask.type(Tensor))
                gnd = Variable(im_gnd.type(Tensor))

            pred = model(im_u, k_u, mask)
            err = criterion(pred, gnd)
            
            for im_gnd, im_und, pred in zip(im,
                                            from_tensor_format(im_und.numpy()),
                                            from_tensor_format(pred.data.cpu().numpy())):
                valid_base_psnr += complex_psnr(im_gnd, im_und, peak='max')
                valid_psnr += complex_psnr(im_gnd, pred, peak='max')
                
            if err < best_valid_loss:
                best_valid_loss = err
                torch.save(model.state_dict(), 'best-model_l5_epochs50.pt')

            validate_err += err.item()
            validate_batches += 1
        epoch_end = time.time()
        validPSNR.append(valid_psnr/validate_batches)
        validBasePSNR.append(valid_base_psnr/validate_batches)
        validERROR.append(validate_err/validate_batches)
        print('Epoch {} done in {} s'.format(epoch, epoch_end - epoch_start))
    base_psnr = 0
    test_psnr = 0
    test_batches = 0
    vis = []
    # LOAD BEST TRAINED MODEL
    model.load_state_dict(torch.load('best-model_l5_epochs50.pt'))
    model.eval()
    test_err = 0
    for im in iterate_minibatch(test, test_batch_size, shuffle=False):
        im_und, k_und, mask, im_gnd = prep_input(im, acc)
        with torch.no_grad():
            im_u = Variable(im_und.type(Tensor))
            k_u = Variable(k_und.type(Tensor))
            mask = Variable(mask.type(Tensor))
            gnd = Variable(im_gnd.type(Tensor))

        pred = model(im_u, k_u, mask)
        err = criterion(pred, gnd)
        test_err += err.item()
        testERROR.append(test_err)

        
        for im_i, und_i, pred_i in zip(im,
                                    from_tensor_format(im_und.numpy()), # (#,2,256,32,30)-> (2,30,256,32)
                                    from_tensor_format(pred.data.cpu().numpy())):
            base_psnr += complex_psnr(im_i, und_i, peak='max')
            test_psnr += complex_psnr(im_i, pred_i, peak='max')

        if save_fig and test_batches % save_every == 0:
            vis.append((from_tensor_format(im_gnd.numpy())[0],
                        from_tensor_format(pred.data.cpu().numpy())[0],
                        from_tensor_format(im_und.numpy())[0],
                        from_tensor_format(mask.data.cpu().numpy(), mask=True)[0]))

        test_batches += 1
        if debug_mode and test_batches == 20:
            break

    t_end = time.time()

    train_err /= train_batches
    validate_err /= validate_batches
    test_err /= test_batches
    base_psnr /= (test_batches * test_batch_size)
    test_psnr /= (test_batches * test_batch_size)

    print('number of epochs is {} and accelerations is {}'.format(num_epoch,acc))
    # Then we print the results for this epoch:
    print("Epoch {}/{}".format(epoch + 1, num_epoch))
    print(" time: {}s".format(t_end - t_start))
    print(" training loss:\t\t{:.6f}".format(train_err))
    print(" validation loss:\t{:.6f}".format(validate_err))
    print(" test loss:\t\t{:.6f}".format(test_err))
    print(" base PSNR:\t\t{:.6f}".format(base_psnr))
    print(" test PSNR:\t\t{:.6f}".format(test_psnr))
    my_dict = {}
    my_dict['train'] = trainERROR
    my_dict['validation'] = validERROR
    my_dict['test'] = test_err
    my_dict['trainBasePSNR'] = trainBasePSNR
    my_dict['trainPSNR'] = trainPSNR
    my_dict['validBasePSNR'] = validBasePSNR
    my_dict['validPSNR'] = validPSNR
    my_dict['testBasePSNR'] = base_psnr
    my_dict['testPSNR'] = test_psnr
    pikla(my_dict, 'dictionary-lambda10-acc{}'.format(int(acc)))
    end = time.time()
    print('ended training with acc {}, in {}, and saved file.'.format(acc,end-start))

    fig,ax = plt.subplots(figsize = (15,7))
    ax.plot(range(1, len(trainERROR)+1), trainERROR, label='Train Loss')
    ax.plot(range(1, len(validERROR)+1), validERROR, label='Validation Loss')
    ax.legend()
    plt.title('Train/Validation Loss vs Epochs (with acceleration: {})'.format(acc))
    os.chdir('/home/nick/bioFinal/fixedLambda/')
    fig.savefig('Lambda10-acceleration{}.png'.format(int(acc)))
    os.chdir('/home/nick/Biodata/DATA')
    pikla(vis, 'vis-lambda10-acc{}'.format(acc))
