import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from os.path import join
from numpy.lib.stride_tricks import as_strided
from numpy.fft import fft, fft2, ifft2, ifft, ifftshift, fftshift



##################################### METRIC BLOCK #####################################################################

def mse(x, y):
    return np.mean(np.abs(x - y)**2)


def psnr(x, y):
    '''
    Measures the PSNR of recon w.r.t x.
    Image must be of either integer (0, 256) or float value (0,1)
    :param x: [m,n]
    :param y: [m,n]
    :return:
    '''
    assert x.shape == y.shape
    assert x.dtype == y.dtype or np.issubdtype(x.dtype, np.float) \
        and np.issubdtype(y.dtype, np.float)
    if x.dtype == np.uint8:
        max_intensity = 256
    else:
        max_intensity = 1

    mse = np.sum((x - y) ** 2).astype(float) / x.size
    return 20 * np.log10(max_intensity) - 10 * np.log10(mse)


def complex_psnr(x, y, peak='normalized'):
    '''
    x: reference image
    y: reconstructed image
    peak: normalised or max

    Notice that ``abs'' squares
    Be careful with the order, since peak intensity is taken from the reference
    image (taking from reconstruction yields a different value).

    '''
    mse = np.mean(np.abs(x - y)**2)
    if peak == 'max':
        return 10*np.log10(np.max(np.abs(x))**2/mse)
    else:
        return 10*np.log10(1./mse)

#######################################################################################################################


############################################# COMPRESSED SENSING BLOCK ################################################

def fftc(x, axis=-1, norm='ortho'):
    ''' expect x as m*n matrix '''
    return fftshift(fft(ifftshift(x, axes=axis), axis=axis, norm=norm), axes=axis)


def ifftc(x, axis=-1, norm='ortho'):
    ''' expect x as m*n matrix '''
    return fftshift(ifft(ifftshift(x, axes=axis), axis=axis, norm=norm), axes=axis)


def fft2c(x, norm='ortho'):
    '''
    Centered fft
    Note: fft2 applies fft to last 2 axes by default
    :param x: 2D onwards. e.g: if its 3d, x.shape = (n,row,col). 4d:x.shape = (n,slice,row,col)
    :return:
    '''
    # axes = (len(x.shape)-2, len(x.shape)-1)  # get last 2 axes
    axes = (-2, -1)  # get last 2 axes
    res = fftshift(fft2(ifftshift(x, axes=axes), norm='ortho'), axes=axes)
    return res


def ifft2c(x, norm='ortho'):
    '''
    Centered ifft
    Note: fft2 applies fft to last 2 axes by default
    :param x: 2D onwards. e.g: if its 3d, x.shape = (n,row,col). 4d:x.shape = (n,slice,row,col)
    :return:
    '''
    axes = (-2, -1)  # get last 2 axes
    res = fftshift(ifft2(ifftshift(x, axes=axes), norm='ortho'), axes=axes)
    return res

def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)

def cartesian_mask(shape, acc, sample_n=10, centred=False):
    """
    Sampling density estimated from implementation of kt FOCUSS

    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..

    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]# 10, 256, 64
    pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1./Nx

    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = ifftshift(mask, axes=(-1, -2))

    return mask

def undersample(x, mask, centred=False, norm='ortho', noise=0):
    '''
    Undersample x. FFT2 will be applied to the last 2 axis
    Parameters
    ----------
    x: array_like
        data
    mask: array_like
        undersampling mask in fourier domain

    norm: 'ortho' or None
        if 'ortho', performs unitary transform, otherwise normal dft

    noise_power: float
        simulates acquisition noise, complex AWG noise.
        must be percentage of the peak signal

    Returns
    -------
    xu: array_like
        undersampled image in image domain. Note that it is complex valued

    x_fu: array_like
        undersampled data in k-space

    '''
    assert x.shape == mask.shape # For Debugging Checks if Inputs in the Right(same) dimensions. 
    
    # zero mean complex Gaussian noise
    noise_power =noise
    nz = np.sqrt(.5)*(np.random.normal(0, 1, x.shape) + 1j * np.random.normal(0, 1, x.shape))
    nz = nz * np.sqrt(noise_power)

    if norm == 'ortho':
        # multiplicative factor
        nz = nz * np.sqrt(np.prod(mask.shape[-2:]))
    else:
        nz = nz * np.prod(mask.shape[-2:])

    if centred:
        x_f = fft2c(x, norm=norm)
        x_fu = mask * (x_f + nz)
        x_u = ifft2c(x_fu, norm=norm)
        return x_u, x_fu
    else:
        x_f = fft2(x, norm= norm)
        x_fu = mask * (x_f + nz)
        x_u = ifft2(x_fu, norm=norm)
        return x_u, x_fu

def undersampling_rate(mask):
    return float(mask.sum()) / mask.size

########################################################################################################################

####################################### DATA MANIPULATION BLOCK ########################################################


def r2c(x, axis=1):
    """Convert pseudo-complex data (2 real channels) to complex data

    x: ndarray
        input data
    axis: int
        the axis that is used to represent the real and complex channel.
        e.g. if axis == i, then x.shape looks like (n_1, n_2, ..., n_i-1, 2, n_i+1, ..., nm)
    """
    shape = x.shape
    if axis < 0: axis = x.ndim + axis
    ctype = np.complex64 if x.dtype == np.float32 else np.complex128

    if axis < len(shape):
        newshape = tuple([i for i in range(0, axis)]) \
                   + tuple([i for i in range(axis+1, x.ndim)]) + (axis,)

        x = x.transpose(newshape)

    x = np.ascontiguousarray(x).view(dtype=ctype)
    return x.reshape(x.shape[:-1])


def c2r(x, axis=1):
    """Convert complex data to pseudo-complex data (2 real channels)

    x: ndarray
        input data
    axis: int
        the axis that is used to represent the real and complex channel.
        e.g. if axis == i, then x.shape looks like (n_1, n_2, ..., n_i-1, 2, n_i+1, ..., nm)
    """
    shape = x.shape
    #dtype = np.float32 if x.dtype == np.complex64 or x.dtype == np.complex128 or x.dtype == np.float32 or x.dtype == np.float64 else np.float64
    dtype = np.float32 or x.dtype == np.float32 or x.dtype == 'float64' if x.dtype == np.complex64  else np.float64
   ## dtype = np.float32 
    
    
    x = np.ascontiguousarray(x).view(dtype=dtype).reshape(shape + (2,))

    n = x.ndim
    if axis < 0: axis = n + axis
    if axis < n:
        newshape = tuple([i for i in range(0, axis)]) + (n-1,) \
                   + tuple([i for i in range(axis, n-1)])
        x = x.transpose(newshape)

    return x

def c2r2(x, axis=1):
    """Convert complex data to pseudo-complex data (2 real channels)

    x: ndarray
        input data
    axis: int
        the axis that is used to represent the real and complex channel.
        e.g. if axis == i, then x.shape looks like (n_1, n_2, ..., n_i-1, 2, n_i+1, ..., nm)
    """
    shape = x.shape
    #dtype = np.float32 if x.dtype == np.complex64 or x.dtype == np.complex128 or x.dtype == np.float32 or x.dtype == np.float64 else np.float64
    #dtype = np.float32 or x.dtype == np.float32 or x.dtype == 'float64' if x.dtype == np.complex64  else np.float64
    dtype = np.float32 
    
    
    x = np.ascontiguousarray(x).view(dtype=dtype).reshape(shape + (2,))

    n = x.ndim
    if axis < 0: axis = n + axis
    if axis < n:
        newshape = tuple([i for i in range(0, axis)]) + (n-1,) \
                   + tuple([i for i in range(axis, n-1)])
        x = x.transpose(newshape)

    return x


def mask_r2c(m):
    return m[0] if m.ndim == 3 else m[:, 0]


def to_tensor_format(x, mask=False):
    """
    Assumes data is of shape (n[, nt], nx, ny).
    Reshapes to (n, n_channels, nx, ny[, nt])
    Note: Depth must be the last axis, the dimensions will be reordered
    """
    if x.ndim == 4:  # n 3D inputs. reorder axes
        x = np.transpose(x, (0, 2, 3, 1))
        
        ##
        if not mask:
            x = c2r(x)
        
    else:
        x = x

    if mask:  # Hacky solution
        x = c2r2(x)
        x = x*(1+1j)

    #x = c2r(x)

    return x


def from_tensor_format(x, mask=False):
    """
    Assumes data is of shape (n, 2, nx, ny[, nt]).
    Reshapes to (n, [nt, ]nx, ny)
    """
    if x.ndim == 5:  # n 3D inputs. reorder axes
        x = np.transpose(x, (0, 1, 4, 2, 3))
    else:
        x = x 

    if mask:
        x = mask_r2c(x)
    else:
        x = r2c(x)

    return x

def prep_input(im, acc=4.0):
    """Undersample the batch, then reformat them into what the network accepts.

    Parameters
    ----------
    gauss_ivar: float - controls the undersampling rate.
                        higher the value, more undersampling
    """
    mask = (cartesian_mask(im.shape, acc, sample_n=8))
    im_und, k_und = undersample(im, mask, centred=False, norm='ortho')
    im_gnd_l = torch.from_numpy(dataToTensor(im))
    im_und_l = torch.from_numpy(dataToTensor(im_und))
    k_und_l = torch.from_numpy(dataToTensor(k_und))
    mask_l = torch.from_numpy(dataToTensor(mask))
    
    # im_gnd_l = torch.from_numpy(to_tensor_format(im))
    # im_und_l = torch.from_numpy(to_tensor_format(im_und))
    # k_und_l = torch.from_numpy(to_tensor_format(k_und))
    # mask_l = torch.from_numpy(to_tensor_format((mask), mask=True))
    
    
    return im_und_l, k_und_l, mask_l, im_gnd_l


def iterate_minibatch(data, batch_size, shuffle=True):
    '''
    data [20, 30, 256, 32] '[#,Nt,Nx,Ny]
    batch_size 2 
    
    Creates a Generator 
    '''
    n = len(data)

    if shuffle:
        data = np.random.permutation(data)

    for i in range(0, n, batch_size):
        yield data[i:i + batch_size]


########################################################################################################################

####################################### DATA GENERATION BLOCK #########################################################

def create_dummy_data():
    """Create small cardiac data based on patches for demo.

    Note that in practice, at test time the method will need to be applied to
    the whole volume. In addition, one would need more data to prevent
    overfitting.

    """
    data = loadmat('cardiac.mat')['seq']
    nx, ny, nt = data.shape
    ny_red = 8
    sl = ny // ny_red
    data_t = np.transpose(data, (2, 0, 1))

    # Synthesize data by extracting patches
    train = np.array([data_t[..., i:i + sl] for i in np.random.randint(0, sl * 3, 20)])
    validate = np.array([data_t[..., i:i + sl] for i in (sl * 4, sl * 5)])
    test = np.array([data_t[..., i:i + sl] for i in (sl * 6, sl * 7)])

    return train, validate, test


def dataToTensor(data):
    data2 = np.zeros((data.shape[0], 2, data.shape[1], data.shape[2]))
    data2[:,0,:,:]= np.real(data)
    data2[:,1,:,:]= np.imag(data)
    return data2 
########################################################################################################################


############################################# MODEL BLOCK ##############################################################

def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)


def relu():
    return nn.ReLU(inplace=True)


def conv_block_dnn(n_ch, nd, nf=32, ks=3, dilation=1, bn=False, nl='lrelu', conv_dim=2, n_out=None):

    # convolution dimension (2D or 3D)
    if conv_dim == 2:
        conv = nn.Conv2d
    else:
        conv = nn.Conv3d

    # output dim: If None, it is assumed to be the same as n_ch
    if not n_out:
        n_out = n_ch

    # dilated convolution
    pad_conv = 1
    if dilation > 1:
        # in = floor(in + 2*pad - dilation * (ks-1) - 1)/stride + 1)
        # pad = dilation
        pad_dilconv = dilation
    else:
        pad_dilconv = pad_conv

    def conv_i():
        return conv(nf,   nf, ks, stride=1, padding=pad_dilconv, dilation=dilation, bias=True)

    conv_1 = conv(n_ch, nf, ks, stride=1, padding=pad_conv, bias=True)
    conv_n = conv(nf, n_out, ks, stride=1, padding=pad_conv, bias=True)

    # relu
    nll = relu if nl == 'relu' else lrelu

    layers = [conv_1, nll()]
    for i in range(nd-2):
        if bn:
            layers += [conv_i(), nll()]

    layers += [conv_n]

    return nn.Sequential(*layers)

def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
    return out


#writing

class DataConsistencyInKspace(nn.Module):
    """ Create data consistency operator

    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.

    """

    def __init__(self, noise_lvl=10.0, norm='ortho'):
        super(DataConsistencyInKspace, self).__init__()
        self.normalized = norm == 'ortho'
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """

        if x.dim() == 4: # input is 2D
            x    = x.permute(0, 2, 3, 1).contiguous()
            k0   = k0.permute(0, 2, 3, 1).contiguous()
            mask = mask.permute(0, 2, 3, 1).contiguous()
        elif x.dim() == 5: # input is 3D
            x    = x.permute(0, 4, 2, 3, 1)
            k0   = k0.permute(0, 4, 2, 3, 1)
            mask = mask.permute(0, 4, 2, 3, 1)

        x_new = torch.view_as_complex(x)
        k = torch.fft.fft2(x_new, norm="ortho")
        k_new = torch.view_as_real(k)
        out = data_consistency(k_new, k0, mask, self.noise_lvl)
        out_new = torch.view_as_complex(out)
        x_res_new = torch.fft.ifft2(out_new, norm="ortho")
        x_res = torch.view_as_real(x_res_new)

        if x.dim() == 4:
            x_res = x_res.permute(0, 3, 1, 2)
        elif x.dim() == 5:
            x_res = x_res.permute(0, 4, 2, 3, 1)

        return x_res


class CS_MRI_MRF_REC(nn.Module):

    def __init__(self, n_channels=2, ns=5, nf=5):
        super(CS_MRI_MRF_REC, self).__init__()
        self.ns = ns
        self.nf = nf
        print('Creating F{}S{}'.format(nf, ns))
        conv_blocks = []
        f_space_blocks = []

        conv_layer = conv_block_dnn

        for i in range(nf):

            conv_blocks.append(conv_layer(n_channels, ns))
            f_space_blocks.append(DataConsistencyInKspace(norm='ortho'))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = f_space_blocks

    def forward(self, x, k, m):
        for i in range(self.ns):
            x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            x = self.dcs[i].perform(x, k, m)

        return x


########################################################################################################################