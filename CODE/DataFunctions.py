import glob
import numpy as np


def DataLoader(data_path, H=256, W=256):
    '''Load All the kSpace files from a given Directory
    and returns the Images in shape
    [#filesm 256,256,2]
     !!!!! data_path ENDING with '/' !!!!!
     '''
    path_all = data_path + '*.npy'
    print(path_all)
    kspace_files = np.asarray(glob.glob(path_all))
    print(kspace_files.shape)

    # Shuffle the Data 
    indexes = np.arange(kspace_files.size, dtype=int)
    np.random.shuffle(indexes)
    kspace_files = kspace_files[indexes]
    print(len(kspace_files))

    nfiles = 0
    for ii in range(len(kspace_files)):
        nfiles += np.load(kspace_files[ii]).shape[0]

    images = np.zeros((nfiles, H, W, 2))
    aux_counter = 0
    norm = np.sqrt(H * W)

    for ii in range(len(kspace_files)):
        aux_kspace = np.load(kspace_files[ii]) / norm
        aux = int(aux_kspace.shape[0])
        aux2 = np.fft.ifft2(aux_kspace[:, :, :, 0] + 1j * aux_kspace[:, :, :, 1])

        images[aux_counter:aux_counter + aux, :, :, 0] = aux2.real
        images[aux_counter:aux_counter + aux, :, :, 1] = aux2.imag
        aux_counter += aux

    # Shuffle training    
    indexes = np.arange(images.shape[0], dtype=int)
    np.random.shuffle(indexes)
    images = images[indexes]
    print("Number of training samples", images.shape[0], 'Files Shape:', images.shape)

    return images
