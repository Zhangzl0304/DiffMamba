from dnn_io import to_tensor_format
from os import listdir
from os.path import join
import pandas as pd
import torch.utils.data as data
import h5py
import numpy as np
import random
# from cascadenet_pytorch.dnn_io import to_tensor_format
import torch
from numpy.fft import fft, fft2, ifft2, ifft, ifftshift, fftshift
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from numpy.lib.stride_tricks import as_strided
import scipy
from icecream import ic
CenterCrop = transforms.CenterCrop(128)

def fft2c(x):
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


def fft2c_tensor(x):
    '''
    Centered fft
    :param x: Input tensor. Shape: (batch_size, channels, height, width)
    :return: Centered fft of the input tensor
    '''
    # Assuming x is a torch.Tensor
    axes = (-2, -1)  # get last 2 axes
    x_ifftshift = torch.fft.ifftshift(x, dim=axes)
    x_fft = torch.fft.fftn(x_ifftshift, dim=axes)
    res = torch.fft.fftshift(x_fft, dim=axes)
    return res


def ifft2c(x):
    '''
    Centered ifft
    Note: fft2 applies fft to last 2 axes by default
    :param x: 2D onwards. e.g: if its 3d, x.shape = (n,row,col). 4d:x.shape = (n,slice,row,col)
    :return:
    '''
    axes = (-2, -1)  # get last 2 axes
    res = fftshift(ifft2(ifftshift(x, axes=axes), norm='ortho'), axes=axes)
    return res

def ifft2c_tensor(x):
    '''
    Centered ifft
    :param x: Input tensor. Shape: (batch_size, channels, height, width)
    :return: Centered ifft of the input tensor
    '''
    # Assuming x is a torch.Tensor
    axes = (-2, -1)  # get last 2 axes
    x_ifftshift = torch.fft.ifftshift(x, dim=axes)
    x_ifft = torch.fft.ifftn(x_ifftshift, dim=axes)
    res = torch.fft.fftshift(x_ifft, dim=axes)
    return res

def load_mat(file_path:str):
    with h5py.File(file_path, 'r') as f:
        key0 = list(f.keys())[0]

        assert len(list(f.keys())) == 1, "There is more than 1 key in the mat file."
        try:
            dataset = f[key0][:]
        except KeyError:
            print(f'Key Error, options:{f.keys()}')
    if dataset.ndim > 3:
        dataset = dataset["real"] + 1j*dataset["imag"]
    return dataset

def random_list(range1, num):
    l = []
    for i in range(num):
        l.append(np.random.randint(0, range1))
    return l


def norm_01_im(x):
    x = ifft2c(x)
    x = x / np.max(x)
    x = fft2c(x)
    return np.max(x), x

def norm_Im(x):
    norm_cons = torch.cat([x.real, x.imag]).max()
    x = x / norm_cons
    return x


def crop_cmrx(im):
    # input: kt, kx, ky
    if len(im.shape) >= 3:
        kx, ky = im.shape[-2:]
        im_crop = im[..., ky//4:3*ky//4]
    elif len(im.shape) == 2:
        kx, ky = im.shape
        im_crop = im[:, ky//4:3*ky//4]
    return im_crop

def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)

def cartesian_mask(shape, acc, sample_n=10, centred=True):
    """
    Sampling density estimated from implementation of kt FOCUSS

    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..

    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
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



# mask = cartesian_mask((12,128,128),4)
# plt.imshow(mask[0])
# plt.imsave('mask_pic/'+'mask_0.png',mask[0], cmap='gray')
# plt.show()




tsfm = transforms.Compose([
    # transforms.CenterCrop((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
])

def Z_score(im):
    return (im-torch.mean(im))/torch.std(im)


def Norm_01(im):
    return im/torch.max(im)

def reshap_mask(input, mask):

    b, t, x, y = input.shape
    mask = np.tile(mask[np.newaxis, np.newaxis,...], (b, t, 1, 1))
    return mask

    
    

class TrainDataFullSizeCmaskNorm(data.Dataset):
    def __init__(self, Acc, ax, tsfm, norm):
        super(TrainDataFullSizeCmaskNorm, self).__init__()
        self.data_gnd = '/home/zhenlin/CMRxRecon2023/dataset/SingleCoil/Cine/TrainingSet/FullSample'
        self.patient = [x for x in listdir(self.data_gnd)]
        self.lax = 'cine_{}.mat'.format(ax)
        self.mask = 'cine_{}_mask.mat'.format(ax)
        del_list = []
        for i in range(len(self.patient)):
            if self.lax not in [x for x in listdir(join(self.data_gnd, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        print(len(self.patient), self.patient)
        self.n_suj = len(self.patient)
        self.tsfm = tsfm
        self.acc = Acc
        self.norm = norm


    def __getitem__(self, index):

        dir_gnd = join(self.data_gnd, self.patient[index], self.lax)
        k_gnd = load_mat(dir_gnd)
        im_gnd = ifft2c(k_gnd)

        im_gnd_in = torch.from_numpy(im_gnd)
        # print('org shape',im_gnd_in.shape)
        im_gnd_in = crop_cmrx(im_gnd_in)

        if self.norm == 'Z':
            im_gnd_in = Z_score(im_gnd_in)
        if self.norm == '01':
            im_gnd_in = norm_Im(im_gnd_in)
        
        cropped_k = fft2c(im_gnd_in)
        
        t,s,x,y = im_gnd_in.shape
        mask = cartesian_mask((t,s,x,y), acc = self.acc)
        mask = np.tile(mask[:,np.newaxis, ...], (1, 2, 1, 1, 1))
        undersampled_k = torch.from_numpy(cropped_k) * mask[:,0,...]

                
        im_gnd_in = torch.from_numpy(to_tensor_format(im_gnd_in))
        im_und_in = torch.from_numpy(to_tensor_format(ifft2c(undersampled_k)))
        mask = torch.from_numpy(mask)

        if self.tsfm:
           im_cat = torch.cat((im_gnd_in, im_und_in, mask), dim=1)
           im_cat_in = self.tsfm(im_cat)
           im_gnd_in = im_cat_in[:,0:2,...]
           im_und_in = im_cat_in[:,2:4,...]
           mask = im_cat_in[:,4:6,...]

        for i in range(s):
            im_gnd_out = im_gnd_in[:,:,i,...].float()
            im_und_out = im_und_in[:,:,i,...].float()
            mask_out1 = mask[:,:,i,...]
            yield im_und_out, im_gnd_out, mask_out1

    def __len__(self):
        return self.n_suj
    
    
class TrainDataFullSizeCmaskNormPatch(data.Dataset):
    def __init__(self, Acc, ax, tsfm, norm, patch = 96):
        super(TrainDataFullSizeCmaskNormPatch, self).__init__()
        self.data_gnd = '/home/zhenlin/CMRxRecon2023/dataset/SingleCoil/Cine/TrainingSet/FullSample'
        self.patient = [x for x in listdir(self.data_gnd)]
        self.lax = 'cine_{}.mat'.format(ax)
        # self.mask = 'cine_{}_mask.mat'.format(ax)
        del_list = []
        for i in range(len(self.patient)):
            if self.lax not in [x for x in listdir(join(self.data_gnd, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        random.shuffle(self.patient)
        print(len(self.patient), self.patient)
        self.n_suj = len(self.patient)
        self.tsfm = tsfm
        self.acc = Acc
        self.norm = norm
        self.patch = patch

    def __getitem__(self, index):

        dir_gnd = join(self.data_gnd, self.patient[index], self.lax)
        k_gnd = load_mat(dir_gnd)
        im_gnd = ifft2c(k_gnd)

        im_gnd_in = torch.from_numpy(im_gnd)
        im_gnd_in = crop_cmrx(im_gnd_in)

        if self.norm == 'Z':
            im_gnd_in = Z_score(im_gnd_in)
        if self.norm == '01':
            im_gnd_in = norm_Im(im_gnd_in)
        
        cropped_k = fft2c(im_gnd_in)
        

        t,s,x,y = im_gnd_in.shape
        mask = cartesian_mask((t,s,x,y), acc = self.acc)
        # plt.imshow(mask[0,0,...], cmap='gray')
        # plt.imsave('mask.png',mask[0,0,...], cmap='gray')
        # plt.show()
        mask = np.tile(mask[:,np.newaxis, ...], (1, 2, 1, 1, 1))
        undersampled_k = torch.from_numpy(cropped_k) * mask[:,0,...]
        
        randseed = random.randint(0, y-self.patch)
        # print('randseed',randseed)
                
        im_gnd_in = torch.from_numpy(to_tensor_format(im_gnd_in))
        im_und_in = torch.from_numpy(to_tensor_format(ifft2c(undersampled_k)))
        im_gnd_in = im_gnd_in[...,randseed:randseed+self.patch]
        im_und_in = im_und_in[...,randseed:randseed+self.patch]

        if self.tsfm:
           im_cat = torch.cat((im_gnd_in, im_und_in), dim=1)
           im_cat_in = self.tsfm(im_cat)
           im_gnd_in = im_cat_in[:,0:2,...]
           im_und_in = im_cat_in[:,2:4,...]

        print('inputshape',im_und_in.shape)
        for i in range(s):
            im_gnd_out = im_gnd_in[:,:,i,...].float()
            im_und_out = im_und_in[:,:,i,...].float()
            # print('outputshapeee',im_und_out.shape, im_gnd_out.shape)
            yield im_und_out, im_gnd_out

    def __len__(self):
        return self.n_suj


class TrainDataPadCmaskNormPatch(data.Dataset):
    def __init__(self, Acc, ax, tsfm, norm, patch = 256):
        super(TrainDataPadCmaskNormPatch, self).__init__()
        self.data_gnd = '/home/zhenlin/CMRxRecon2023/dataset/SingleCoil/Cine/TrainingSet/FullSample'
        self.patient = [x for x in listdir(self.data_gnd)]
        self.lax = 'cine_{}.mat'.format(ax)
        # self.mask = 'cine_{}_mask.mat'.format(ax)
        del_list = []
        for i in range(len(self.patient)):
            if self.lax not in [x for x in listdir(join(self.data_gnd, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        # random.shuffle(self.patient)
        print(len(self.patient), self.patient)
        self.n_suj = len(self.patient)
        self.tsfm = tsfm
        self.acc = Acc
        self.norm = norm
        self.patch = patch

    def __getitem__(self, index):

        dir_gnd = join(self.data_gnd, self.patient[index], self.lax)
        k_gnd = load_mat(dir_gnd)
        im_gnd = ifft2c(k_gnd)

        im_gnd_in = torch.from_numpy(im_gnd)
        # print('org shape',im_gnd_in.shape)
        im_gnd_in = crop_cmrx(im_gnd_in)

        if self.norm == 'Z':
            im_gnd_in = Z_score(im_gnd_in)
        if self.norm == '01':
            im_gnd_in = norm_Im(im_gnd_in)
        
        cropped_k = fft2c(im_gnd_in)
        

        t,s,x,y = im_gnd_in.shape
        mask = cartesian_mask((t,s,x,y), acc = self.acc)
        mask = np.tile(mask[:,np.newaxis, ...], (1, 2, 1, 1, 1))
        undersampled_k = torch.from_numpy(cropped_k) * mask[:,0,...]
        
        randseed = random.randint(0, y-self.patch)
        # print('randseed',randseed)
                
        im_gnd_in = torch.from_numpy(to_tensor_format(im_gnd_in))
        im_und_in = torch.from_numpy(to_tensor_format(ifft2c(undersampled_k)))
        im_gnd_in = im_gnd_in[...,randseed:randseed+self.patch]
        im_und_in = im_und_in[...,randseed:randseed+self.patch]

        if self.tsfm:
           im_cat = torch.cat((im_gnd_in, im_und_in), dim=1)
           im_cat_in = self.tsfm(im_cat)
           im_gnd_in = im_cat_in[:,0:2,...]
           im_und_in = im_cat_in[:,2:4,...]

        print('inputshape',im_und_in.shape)
        for i in range(s):
            im_gnd_out = im_gnd_in[:,:,i,...].float()
            im_und_out = im_und_in[:,:,i,...].float()
            # print('outputshapeee',im_und_out.shape, im_gnd_out.shape)
            yield im_und_out, im_gnd_out

    def __len__(self):
        return self.n_suj


class DataLoaderForVis(data.Dataset):
    def __init__(self, Acc, ax, norm):
        super(DataLoaderForVis, self).__init__()
        self.data_gnd = '/home/zhenlin/CMRxRecon2023/dataset/SingleCoil/Cine/TrainingSet/FullSample'
        self.patient = [x for x in listdir(self.data_gnd)]
        self.lax = 'cine_{}.mat'.format(ax)
        # self.mask = 'cine_{}_mask.mat'.format(ax)
        del_list = []
        for i in range(len(self.patient)):
            if self.lax not in [x for x in listdir(join(self.data_gnd, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        # random.shuffle(self.patient)
        print(len(self.patient), self.patient)
        self.n_suj = len(self.patient)
        self.tsfm = tsfm
        self.acc = Acc
        self.norm = norm

    def __getitem__(self, index):

        dir_gnd = join(self.data_gnd, self.patient[index], self.lax)
        k_gnd = load_mat(dir_gnd)
        im_gnd = ifft2c(k_gnd)

        im_gnd_in = torch.from_numpy(im_gnd)
        # print('org shape',im_gnd_in.shape)
        # im_gnd_in = crop_cmrx(im_gnd_in)

        if self.norm == 'Z':
            im_gnd_in = Z_score(im_gnd_in)
        if self.norm == '01':
            im_gnd_in = norm_Im(im_gnd_in)
        
        cropped_k = fft2c(im_gnd_in)
        

        t,s,x,y = im_gnd_in.shape
        mask = cartesian_mask((t,s,x,y), acc = self.acc)
        mask = np.tile(mask[:,np.newaxis, ...], (1, 2, 1, 1, 1))
        undersampled_k = torch.from_numpy(cropped_k) * mask[:,0,...]
        
        # randseed = random.randint(0, y-self.patch)
        # print('randseed',randseed)
                
        im_gnd_in = torch.from_numpy(to_tensor_format(im_gnd_in))
        im_und_in = torch.from_numpy(to_tensor_format(ifft2c(undersampled_k)))

        print('inputshape',im_und_in.shape)
        for i in range(s):
            im_gnd_out = im_gnd_in[:,:,i,...].float()
            im_und_out = im_und_in[:,:,i,...].float()
            # print('outputshapeee',im_und_out.shape, im_gnd_out.shape)
            mask_out = mask[:,:,i,...]
            yield im_und_out, im_gnd_out,mask_out

    def __len__(self):
        return self.n_suj


class TrainDataFullSizeCmaskNorm2(data.Dataset):
    def __init__(self, Acc, ax, tsfm, norm):
        super(TrainDataFullSizeCmaskNorm2, self).__init__()
        self.data_gnd = '/home/zhenlin/CMRxRecon2023/dataset/SingleCoil/Cine/TrainingSet/FullSample'
        self.patient = [x for x in listdir(self.data_gnd)]
        self.lax = 'cine_{}.mat'.format(ax)
        self.mask = 'cine_{}_mask.mat'.format(ax)
        del_list = []
        for i in range(len(self.patient)):
            if self.lax not in [x for x in listdir(join(self.data_gnd, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        print(len(self.patient), self.patient)
        self.n_suj = len(self.patient)
        self.tsfm = tsfm
        self.acc = Acc
        self.norm = norm


    def __getitem__(self, index):

        dir_gnd = join(self.data_gnd, self.patient[index], self.lax)
        k_gnd = load_mat(dir_gnd)
        im_gnd = ifft2c(k_gnd)

        im_gnd_in = torch.from_numpy(im_gnd)
        print('org shape',im_gnd_in.shape)
        im_gnd_in = crop_cmrx(im_gnd_in)
        cropped_k = fft2c(im_gnd_in)
        
        t,s,x,y = im_gnd_in.shape
        mask = cartesian_mask((t,s,x,y), acc = self.acc)
        mask = np.tile(mask[:,np.newaxis, ...], (1, 2, 1, 1, 1))
        undersampled_k = torch.from_numpy(cropped_k) * mask[:,0,...]

                
        im_gnd_in = torch.from_numpy(to_tensor_format(im_gnd_in))
        im_und_in = torch.from_numpy(to_tensor_format(ifft2c(undersampled_k)))
        mask = torch.from_numpy(mask)

        if self.norm == 'Z':
            im_gnd_in = Z_score(im_gnd_in)
            im_und_in = Z_score(im_und_in)
        if self.norm == '01':
            im_gnd_in = im_gnd_in/torch.max(im_gnd_in)
            im_und_in = im_und_in/torch.max(im_gnd_in)

        if self.tsfm:
           im_cat = torch.cat((im_gnd_in, im_und_in, mask), dim=1)
           im_cat_in = self.tsfm(im_cat)
           im_gnd_in = im_cat_in[:,0:2,...]
           im_und_in = im_cat_in[:,2:4,...]
           mask = im_cat_in[:,4:6,...]

        for i in range(s):
            im_gnd_out = im_gnd_in[:,:,i,...].float()
            im_und_out = im_und_in[:,:,i,...].float()
            mask_out1 = mask[:,:,i,...]
            yield im_und_out, im_gnd_out, mask_out1

    def __len__(self):
        return self.n_suj



class TestDataFolderCmask(data.Dataset):
    def __init__(self, Acc, ax, norm):
        super(TestDataFolderCmask, self).__init__()
        self.data_gnd = '/rds/general/user/zz8523/home/test/Cine/TestSet/FullSample'
        self.patient = [x for x in listdir(self.data_gnd)]
        self.mask_path = "/rds/general/user/zz8523/home/test/Cine/TestSet/Cmask"
        self.norm = norm_01_im
        self.lax = 'cine_{}.mat'.format(ax)
        self.mask = 'mask_sax_ACS10.mat'.format(ax)
        del_list = []
        for i in range(len(self.patient)):
            if self.lax not in [x for x in listdir(join(self.data_gnd, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        self.patient = sorted(self.patient)
        print(len(self.patient), self.patient)
        self.n_suj = len(self.patient)
        self.norm = norm

    def __getitem__(self, index):

        dir_gnd = join(self.data_gnd, self.patient[index], self.lax)
        dir_mask = join(self.mask_path, self.patient[index], "Acc8",self.mask)
        mask = scipy.io.loadmat(dir_mask)['data']
        k_gnd = load_mat(dir_gnd)
        im_gnd = ifft2c(k_gnd)
        

        t,s,x,y = im_gnd.shape
        self.slicenum = s
        # print(im_gnd.shape)

        im_gnd_in = torch.from_numpy(im_gnd)
        im_gnd_in = CenterCrop(im_gnd_in)
        cropped_k = fft2c(im_gnd_in)

    
        undersampled_k = torch.from_numpy(cropped_k) * mask
     
        im_gnd_in = torch.from_numpy(to_tensor_format(im_gnd_in))
        im_und_in = torch.from_numpy(to_tensor_format(ifft2c(undersampled_k)))
        mask = np.tile(mask[:,np.newaxis, ...], (1, 2, 1, 1, 1))
        mask = torch.from_numpy(mask)

        if self.norm == 'Z':
            im_gnd_in = Z_score(im_gnd_in)
            im_und_in = Z_score(im_und_in)
        if self.norm == '01':
            im_gnd_in = Norm_01(im_gnd_in)
            im_und_in = Norm_01(im_und_in)

        for i in range(s):
            im_gnd_out = im_gnd_in[:,:,i,...]
            im_und_out = im_und_in[:,:,i,...]
            mask_out = mask[:,:,i,...]
            yield im_und_out, im_gnd_out, mask_out
        
    def __len__(self):
        return self.n_suj



class TestDataFullSizeCmaskNorm(data.Dataset):
    def __init__(self, Acc, ax, norm):
        super(TestDataFullSizeCmaskNorm, self).__init__()
        self.data_gnd = '/bigdata/zhenlin/CMRxRecon2023Data/TestData/Single_test/Cine/TestSet/FullSample'
        self.patient = [x for x in listdir(self.data_gnd)]
        self.mask_path = "/bigdata/zhenlin/CMRxRecon2023Data/TestData/Single_test/Cine/TestSet/Cmask"
        self.lax = 'cine_{}.mat'.format(ax)
        self.mask = 'mask_sax_ACS10.mat'.format(ax)
        del_list = []
        for i in range(len(self.patient)):
            if self.lax not in [x for x in listdir(join(self.data_gnd, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        self.patient = sorted(self.patient)
        print(len(self.patient), self.patient)
        self.n_suj = len(self.patient)
        self.norm = norm
        self.Acc = Acc

    def __getitem__(self, index):

        dir_gnd = join(self.data_gnd, self.patient[index], self.lax)
        dir_mask = join(self.mask_path, self.patient[index], f"Acc{self.Acc}",self.mask)
        mask = scipy.io.loadmat(dir_mask)['data']
        k_gnd = load_mat(dir_gnd)
        im_gnd = ifft2c(k_gnd)
        

        t,s,x,y = im_gnd.shape
        self.slicenum = s
        print(im_gnd.shape)

        im_gnd_in = torch.from_numpy(im_gnd)
        im_gnd_in = crop_cmrx(im_gnd_in)
        if self.norm == 'Z':
            im_gnd_in = Z_score(im_gnd_in)
        if self.norm == '01':
            im_gnd_in = norm_Im(im_gnd_in)    
             
        cropped_k = fft2c(im_gnd_in)
    
        undersampled_k = torch.from_numpy(cropped_k) * mask
     
        im_gnd_in = torch.from_numpy(to_tensor_format(im_gnd_in))
        im_und_in = torch.from_numpy(to_tensor_format(ifft2c(undersampled_k)))
        mask = np.tile(mask[:,np.newaxis, ...], (1, 2, 1, 1, 1))
        mask = torch.from_numpy(mask)

        for i in range(s):
            im_gnd_out = im_gnd_in[:,:,i,...]
            im_und_out = im_und_in[:,:,i,...]
            mask_out = mask[:,:,i,...]
            yield im_und_out, im_gnd_out, mask_out
        
    def __len__(self):
        return self.n_suj



class TestCmaskNormMultiOneCoil(data.Dataset):
    def __init__(self, Acc, ax, norm):
        super(TestCmaskNormMultiOneCoil, self).__init__()
        self.data_gnd = '/bigdata/zhenlin/CMRxRecon2023Data/TestData/Multi_test/MultiCoil/Cine/TestSet/FullSample'
        self.patient = [x for x in listdir(self.data_gnd)]
        self.mask_path = "/bigdata/zhenlin/CMRxRecon2023Data/TestData/Single_test/Cine/TestSet/Cmask"    
        self.ax = 'cine_{}.mat'.format(ax)
        self.mask = 'mask_sax_ACS10.mat'.format(ax)
        del_list = []
        for i in range(len(self.patient)):
            if self.ax not in [x for x in listdir(join(self.data_gnd, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        # self.all_data_gnd = [item for item in self.data_gnd for _ in range(10)]
        # self.all_mask_path = [item for item in self.mask_path for _ in range(10)]
        # self.all_sens_path = [item for item in self.sens_path for _ in range(10)]
        self.patient = sorted(self.patient)
        self.all_patient = [item for item in self.patient for _ in range(10)]
        print(len(self.all_patient), self.all_patient)
        self.n_suj = len(self.all_patient)
        self.norm = norm
        self.Acc = Acc

    def __getitem__(self, index):

        dir_gnd = join(self.data_gnd, self.all_patient[index], self.ax)
        dir_mask = join(self.mask_path, self.all_patient[index], f"Acc{self.Acc}",self.mask)
        # dir_sens_map = join(self.sens_path, self.all_patient[index], self.sens_map)
        
        mask = scipy.io.loadmat(dir_mask)['data']
        k_gnd = load_mat(dir_gnd)
        im_gnd_in = torch.from_numpy(ifft2c(k_gnd))
        im_gnd_coil = im_gnd_in[:,:,index%10,...]
        print('imgndshape',im_gnd_coil.shape, 'sens',mask.shape)

        im_gnd_in = crop_cmrx(im_gnd_coil)
        
        if self.norm == 'Z':
            im_gnd_in = Z_score(im_gnd_in)
        if self.norm == '01':
            im_gnd_in = norm_Im(im_gnd_in)    
             
        cropped_k = fft2c(im_gnd_in)
    
        undersampled_k = torch.from_numpy(cropped_k) * mask
     
        im_gnd_in = torch.from_numpy(to_tensor_format(im_gnd_in))
        im_und_in = torch.from_numpy(to_tensor_format(ifft2c(undersampled_k)))
        mask = np.tile(mask[:,np.newaxis, ...], (1, 2, 1, 1, 1))
        mask = torch.from_numpy(mask)
        t,c,s,h,w = im_gnd_in.shape

        for i in range(s):
            im_gnd_out = im_gnd_in[:,:,i,...]
            im_und_out = im_und_in[:,:,i,...]
            mask_out = mask[:,:,i,...]
            yield im_und_out, im_gnd_out, mask_out
        
    def __len__(self):
        return self.n_suj


class TrainCmaskNormMultiOneCoil(data.Dataset):
    def __init__(self, Acc, ax, norm, patch):
        super(TrainCmaskNormMultiOneCoil, self).__init__()
        self.data_gnd = '/bigdata/zhenlin/CMRxRecon2023Data/ChallengeData/MultiCoil/Cine/TrainingSet/FullSample'
        self.patient = [x for x in listdir(self.data_gnd)]
        self.ax = 'cine_{}.mat'.format(ax)
        del_list = []
        for i in range(len(self.patient)):
            if self.ax not in [x for x in listdir(join(self.data_gnd, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        # random.shuffle(self.patient)
        
        self.all_patient = [item for item in self.patient for _ in range(10)]
        print(len(self.all_patient), self.all_patient)
        self.n_suj = len(self.all_patient)
        self.tsfm = False
        self.acc = Acc
        self.norm = norm
        self.patch = patch

    def __getitem__(self, index):

        dir_gnd = join(self.data_gnd, self.all_patient[index], self.ax)
        k_gnd = load_mat(dir_gnd)
        im_gnd = ifft2c(k_gnd)

        im_gnd_in = torch.from_numpy(im_gnd)
        im_gnd_in = im_gnd_in[:,:,index%10,...]
        im_gnd_in = crop_cmrx(im_gnd_in)

        if self.norm == 'Z':
            im_gnd_in = Z_score(im_gnd_in)
        if self.norm == '01':
            im_gnd_in = norm_Im(im_gnd_in)
            
        
        cropped_k = fft2c(im_gnd_in)
        

        t,s,x,y = im_gnd_in.shape
        mask = cartesian_mask((t,s,x,y), acc = self.acc)
        mask = np.tile(mask[:,np.newaxis, ...], (1, 2, 1, 1, 1))
        undersampled_k = torch.from_numpy(cropped_k) * mask[:,0,...]
        
        randseed = random.randint(0, y-self.patch)
        # print('randseed',randseed)
                
        im_gnd_in = torch.from_numpy(to_tensor_format(im_gnd_in))
        im_und_in = torch.from_numpy(to_tensor_format(ifft2c(undersampled_k)))
        im_gnd_in = im_gnd_in[...,randseed:randseed+self.patch]
        im_und_in = im_und_in[...,randseed:randseed+self.patch]

        if self.tsfm:
           im_cat = torch.cat((im_gnd_in, im_und_in), dim=1)
           im_cat_in = self.tsfm(im_cat)
           im_gnd_in = im_cat_in[:,0:2,...]
           im_und_in = im_cat_in[:,2:4,...]

        print('inputshape',im_und_in.shape)
        for i in range(s):
            im_gnd_out = im_gnd_in[:,:,i,...].float()
            im_und_out = im_und_in[:,:,i,...].float()
            # print('outputshapeee',im_und_out.shape, im_gnd_out.shape)
            yield im_und_out, im_gnd_out

    def __len__(self):
        return self.n_suj


class TrainDataFullSizeCmaskNormPatchMulti(data.Dataset):
    def __init__(self, Acc, ax, tsfm, norm, patch = 96):
        super(TrainDataFullSizeCmaskNormPatchMulti, self).__init__()
        self.data_gnd = '/bigdata/zhenlin/CMRxRecon2023Data/ChallengeData/MultiCoil/Cine/TrainingSet/FullSample'
        self.patient = [x for x in listdir(self.data_gnd)]
        self.ax = 'cine_{}.mat'.format(ax)
        self.sens_map = f'cine_{ax}_sen.mat'
        self.sens_path = '/bigdata/zhenlin/CMRxRecon2023Data/ChallengeData/Sens/MultiCoil/Cine/TrainingSet/FullSample'
        del_list = []
        for i in range(len(self.patient)):
            if self.ax not in [x for x in listdir(join(self.data_gnd, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        random.shuffle(self.patient)
        print(len(self.patient), self.patient)
        self.n_suj = len(self.patient)
        self.tsfm = tsfm
        self.acc = Acc
        self.norm = norm
        self.patch = patch

    def __getitem__(self, index):

        dir_gnd = join(self.data_gnd, self.patient[index], self.ax)
        dir_sens_map = join(self.sens_path, self.patient[index], self.sens_map)
        k_gnd = load_mat(dir_gnd)
        
        sens_map = scipy.io.loadmat(dir_sens_map)['SNS_sz']
        sens_map = torch.from_numpy(np.transpose(sens_map, (3,2,1,0)))  
   
        im_gnd = ifft2c(k_gnd)       

        im_gnd_in = torch.from_numpy(im_gnd)
        weighted_images = im_gnd_in * sens_map.conj()  
        
        combined_image = torch.sum(weighted_images, axis=2)
               

        sensitivity_sum = torch.sum(sens_map * sens_map.conj(), axis=1)
        final_image = combined_image / (sensitivity_sum + 1e-8)
        im_gnd_in = crop_cmrx(final_image)

        if self.norm == 'Z':
            im_gnd_in = Z_score(im_gnd_in)
        if self.norm == '01':
            im_gnd_in = norm_Im(im_gnd_in)
        
        cropped_k = fft2c(im_gnd_in)
        

        t,s,x,y = im_gnd_in.shape
        mask = cartesian_mask((t,s,x,y), acc = self.acc)
        # plt.imshow(mask[0,0,...], cmap='gray')
        # plt.imsave('mask.png',mask[0,0,...], cmap='gray')
        # plt.show()
        mask = np.tile(mask[:,np.newaxis, ...], (1, 2, 1, 1, 1))
        undersampled_k = torch.from_numpy(cropped_k) * mask[:,0,...]
        
        randseed = random.randint(0, y-self.patch)

                
        im_gnd_in = torch.from_numpy(to_tensor_format(im_gnd_in))
        im_und_in = torch.from_numpy(to_tensor_format(ifft2c(undersampled_k)))
        im_gnd_in = im_gnd_in[...,randseed:randseed+self.patch]
        im_und_in = im_und_in[...,randseed:randseed+self.patch]

        if self.tsfm:
           im_cat = torch.cat((im_gnd_in, im_und_in), dim=1)
           im_cat_in = self.tsfm(im_cat)
           im_gnd_in = im_cat_in[:,0:2,...]
           im_und_in = im_cat_in[:,2:4,...]

        print('inputshape',im_und_in.shape)
        for i in range(s):
            im_gnd_out = im_gnd_in[:,:,i,...].float()
            im_und_out = im_und_in[:,:,i,...].float()
            yield im_und_out, im_gnd_out

    def __len__(self):
        return self.n_suj



class TestDataFullSizeCmaskNormPatchMulti(data.Dataset):
    def __init__(self, Acc, ax, tsfm, norm, patch = 96):
        super(TestDataFullSizeCmaskNormPatchMulti, self).__init__()
        self.data_gnd = '/bigdata/zhenlin/CMRxRecon2023Data/TestData/Multi_test/MultiCoil/Cine/TestSet/FullSample'
        self.patient = [x for x in listdir(self.data_gnd)]
        self.ax = 'cine_{}.mat'.format(ax)
        self.sens_map = f'cine_{ax}_sen.mat'
        self.sens_path = '/bigdata/zhenlin/CMRxRecon2023Data/ChallengeData/Sens/MultiCoil/Cine/TrainingSet/FullSample'
        del_list = []
        for i in range(len(self.patient)):
            if self.ax not in [x for x in listdir(join(self.data_gnd, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        random.shuffle(self.patient)
        print(len(self.patient), self.patient)
        self.n_suj = len(self.patient)
        self.tsfm = tsfm
        self.acc = Acc
        self.norm = norm
        self.patch = patch

    def __getitem__(self, index):

        dir_gnd = join(self.data_gnd, self.patient[index], self.ax)
        dir_sens_map = join(self.sens_path, self.patient[index], self.sens_map)
        k_gnd = load_mat(dir_gnd)
        
        sens_map = scipy.io.loadmat(dir_sens_map)['SNS_sz']
        sens_map = torch.from_numpy(np.transpose(sens_map, (3,2,1,0)))  
   
        im_gnd = ifft2c(k_gnd)       

        im_gnd_in = torch.from_numpy(im_gnd)
        weighted_images = im_gnd_in * sens_map.conj()  
        
        combined_image = torch.sum(weighted_images, axis=2)
               

        sensitivity_sum = torch.sum(sens_map * sens_map.conj(), axis=1)
        final_image = combined_image / (sensitivity_sum + 1e-8)
        im_gnd_in = crop_cmrx(final_image)

        if self.norm == 'Z':
            im_gnd_in = Z_score(im_gnd_in)
        if self.norm == '01':
            im_gnd_in = norm_Im(im_gnd_in)
        
        cropped_k = fft2c(im_gnd_in)
        

        t,s,x,y = im_gnd_in.shape
        mask = cartesian_mask((t,s,x,y), acc = self.acc)
        mask = np.tile(mask[:,np.newaxis, ...], (1, 2, 1, 1, 1))
        undersampled_k = torch.from_numpy(cropped_k) * mask[:,0,...]
        
        randseed = random.randint(0, y-self.patch)

        im_gnd_in = torch.from_numpy(to_tensor_format(im_gnd_in))
        im_und_in = torch.from_numpy(to_tensor_format(ifft2c(undersampled_k)))
        im_gnd_in = im_gnd_in[...,randseed:randseed+self.patch]
        im_und_in = im_und_in[...,randseed:randseed+self.patch]

        if self.tsfm:
           im_cat = torch.cat((im_gnd_in, im_und_in), dim=1)
           im_cat_in = self.tsfm(im_cat)
           im_gnd_in = im_cat_in[:,0:2,...]
           im_und_in = im_cat_in[:,2:4,...]

        print('inputshape',im_und_in.shape)
        for i in range(s):
            im_gnd_out = im_gnd_in[:,:,i,...].float()
            im_und_out = im_und_in[:,:,i,...].float()
            yield im_und_out, im_gnd_out

    def __len__(self):
        return self.n_suj

    

class LoadTrainDatanpzPatchMulti(data.Dataset):
    def __init__(self, Acc, ax, tsfm, patch = 96):
        super(LoadTrainDatanpzPatchMulti, self).__init__()
        self.sens_path = '/bigdata/zhenlin/CMRxRecon2023Data/ChallengeData/Sens/MultiCoil/Cine/TrainingSet/FullSample'
        self.patient = [x for x in listdir(self.sens_path)]
        self.file_name = f'data_{ax}_combined.npz'
        self.sens_map = f'cine_{ax}_sen.mat'
        del_list = []
        for i in range(len(self.patient)):
            if self.sens_map not in [x for x in listdir(join(self.sens_path, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        random.shuffle(self.patient)
        self.patient = sorted(self.patient)
        print(len(self.patient), self.patient)
        self.n_suj = len(self.patient)
        self.tsfm = tsfm
        self.acc = Acc
        self.patch = patch

    def __getitem__(self, index):

        dir_gnd = join(self.sens_path, self.patient[index], self.file_name)
        data_all = np.load(dir_gnd)
        
        dir_sens_map = join(self.sens_path, self.patient[index], self.sens_map)
        sens_map = scipy.io.loadmat(dir_sens_map)['SNS_sz']
        sens_map = torch.from_numpy(np.transpose(sens_map, (3,2,1,0)))  
        sens_map = crop_cmrx(sens_map)
        
        data_gnd, _, _ = data_all['data_gnd'], data_all['data_und'], data_all['mask']
        data_all.close()
        data_gnd = norm_Im(torch.tensor(data_gnd)).numpy()
        print(data_gnd.shape)
        t,s,x,y = data_gnd.shape    
        mask = cartesian_mask((t,s,x,y), acc = self.acc) 
        mask_multi = np.tile(mask[:,:,np.newaxis, ...], (1, 1, 10, 1, 1))
        
        # get split gnd
        data_gnd_sp = np.tile(data_gnd[:,:,np.newaxis, ...], (1, 1, 10, 1, 1))
        split_gnd = torch.tensor(data_gnd_sp) * sens_map

        # get combined und
        split_k = fft2c_tensor(split_gnd)
        undersampled_k = split_k * mask_multi
        undersampled_img = ifft2c_tensor(undersampled_k)      
        weighted_images_und = undersampled_img * sens_map.conj()
        data_und = torch.sum(weighted_images_und, axis=2)       

        # get patch
        randseed = random.randint(0, y-self.patch)

        data_gnd = torch.from_numpy(to_tensor_format(data_gnd))
        data_und = torch.from_numpy(to_tensor_format(data_und))
        # mask = np.tile(mask[:,np.newaxis, ...], (1, 2, 1, 1, 1))
        data_und = data_und[...,randseed:randseed+self.patch]
        data_gnd = data_gnd[...,randseed:randseed+self.patch]

        if self.tsfm:
           im_cat = torch.cat((data_gnd, data_und), dim=1)
           im_cat_in = self.tsfm(im_cat)
           im_gnd_in = im_cat_in[:,0:2,...]
           im_und_in = im_cat_in[:,2:4,...]

        # print('inputshape',im_und_in.shape)
        for i in range(s):
            im_gnd_out = im_gnd_in[:,:,i,...].float()
            im_und_out = im_und_in[:,:,i,...].float()
            yield im_und_out, im_gnd_out

    def __len__(self):
        return self.n_suj




def MRcombine(input, sens_map):
        '''
        input: [t,s,coil,x,y]
        sensmap: [t,s,coil,x,y]
        '''
        input = input * sens_map.conj()
        data_und = torch.sum(input, axis=2)     
    


class LoadTrainDatanpzTest(data.Dataset):
    def __init__(self, Acc, ax, tsfm, patch = 256):
        super(LoadTrainDatanpzTest, self).__init__()
        self.sens_path = '/bigdata/zhenlin/CMRxRecon2023Data/ChallengeData/Sens/MultiCoil/Cine/TrainingSet/FullSample'
        self.patient = [x for x in listdir(self.sens_path)]
        self.file_name = f'data_{ax}_combined.npz'
        self.sens_map = f'cine_{ax}_sen.mat'
        del_list = []
        for i in range(len(self.patient)):
            if self.sens_map not in [x for x in listdir(join(self.sens_path, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        random.shuffle(self.patient)
        self.patient = sorted(self.patient)
        print(len(self.patient), self.patient)
        self.n_suj = len(self.patient)
        self.tsfm = tsfm
        self.acc = Acc
        self.patch = patch

    def __getitem__(self, index):

        dir_gnd = join(self.sens_path, self.patient[index], self.file_name)
        data_all = np.load(dir_gnd)
        
        dir_sens_map = join(self.sens_path, self.patient[index], self.sens_map)
        sens_map = scipy.io.loadmat(dir_sens_map)['SNS_sz']
        sens_map = torch.from_numpy(np.transpose(sens_map, (3,2,1,0)))  
        sens_map = crop_cmrx(sens_map)
        
        data_gnd, data_und_org, _ = data_all['data_gnd'], data_all['data_und'], data_all['mask']
        # print(data_und_org.shape)
        data_gnd = norm_Im(torch.tensor(data_gnd)).numpy()
        print(data_gnd.shape)
        t,s,x,y = data_gnd.shape    
        mask = cartesian_mask((t,s,x,y), acc = self.acc) 
        mask_multi = np.tile(mask[:,:,np.newaxis, ...], (1, 1, 10, 1, 1))
        
        data_gnd_sp = np.tile(data_gnd[:,:,np.newaxis, ...], (1, 1, 10, 1, 1))
        split_gnd = torch.tensor(data_gnd_sp) * sens_map
        
        
        split_k = fft2c_tensor(split_gnd)
        undersampled_k = split_k * mask_multi
        undersampled_img = ifft2c_tensor(undersampled_k)      
          
        weighted_images_und = undersampled_img * sens_map.conj()
        data_und = torch.sum(weighted_images_und, axis=2) 
        
        data_und_sp = np.tile(data_und[:,:,np.newaxis, ...], (1, 1, 10, 1, 1))
        data_und2 = torch.tensor(data_und_sp) * sens_map
        weighted_images_und2 = data_und2 * sens_map.conj()
        data_und2 = torch.sum(weighted_images_und2, axis=2)    
        
        # data_und_sp2 = np.tile(data_gnd[:,:,np.newaxis, ...], (1, 1, 10, 1, 1))
        # data_und2 = torch.tensor(data_und_sp) * sens_map
        # weighted_images_und2 = data_und2 * sens_map.conj()
        # data_und2 = torch.sum(weighted_images_und2, axis=2)          
        
        # print('diffund', (fft2c_tensor(data_und2) - fft2c_tensor(data_und)))   
        # print('kkk', fft2c_tensor(data_und2))
        # print(data_und) 

        randseed = random.randint(0, y-self.patch)

        data_gnd = torch.from_numpy(to_tensor_format(data_gnd))
        data_und = torch.from_numpy(to_tensor_format(data_und))
        
        
        # mask = np.tile(mask[:,np.newaxis, ...], (1, 2, 1, 1, 1))
        data_und = data_und[...,randseed:randseed+self.patch]
        data_gnd = data_gnd[...,randseed:randseed+self.patch]

        if self.tsfm:
           im_cat = torch.cat((data_gnd, data_und), dim=1)
           im_cat_in = self.tsfm(im_cat)
           im_gnd_in = im_cat_in[:,0:2,...]
           im_und_in = im_cat_in[:,2:4,...]

        # print('inputshape',im_und_in.shape)
        for i in range(s):
            im_gnd_out = im_gnd_in[:,:,i,...].float()
            im_und_out = im_und_in[:,:,i,...].float()
            yield im_und_out, im_gnd_out

    def __len__(self):
        return self.n_suj



class LoadTrainDataMulti(data.Dataset):
    def __init__(self, Acc, ax, patch=256):
        super(LoadTrainDataMulti, self).__init__()
        self.data_gnd = '/bigdata/zhenlin/CMRxRecon2023Data/ChallengeData/MultiCoil/Cine/TrainingSet/FullSample'
        self.sens_path = '/bigdata/zhenlin/CMRxRecon2023Data/ChallengeData/Sens/MultiCoil/Cine/TrainingSet/FullSample'
        self.patient = [x for x in listdir(self.sens_path)]
        # self.mask_path = f"/bigdata/zhenlin/CMRxRecon2023Data/TestData/Cmask/Acc{Acc}_ACS10"
        self.sens_map = f'cine_{ax}_sen.mat'
        self.ax = 'cine_{}.mat'.format(ax)
        self.mask = 'mask_sax_ACS10.mat'.format(ax)
        del_list = []
        for i in range(len(self.patient)):
            if self.ax not in [x for x in listdir(join(self.data_gnd, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        self.patient = sorted(self.patient)
        print(len(self.patient), self.patient)
        self.n_suj = len(self.patient)
        self.Acc = Acc
        self.patch = patch

    def __getitem__(self, index):

        dir_gnd = join(self.data_gnd, self.patient[index], self.ax)
        # dir_mask = join(self.mask_path, self.patient[index],self.mask)
        dir_sens_map = join(self.sens_path, self.patient[index], self.sens_map)
        
        # mask = scipy.io.loadmat(dir_mask)['data']
        
        sens_map = scipy.io.loadmat(dir_sens_map)['SNS_sz']
        sens_map = torch.from_numpy(np.transpose(sens_map, (3,2,1,0)))
        k_gnd = load_mat(dir_gnd)
        im_gnd_in = torch.from_numpy(ifft2c(k_gnd))
        im_gnd_in = crop_cmrx(im_gnd_in)
        # im_gnd_in = norm_Im(im_gnd_in)
        
        sens_map = crop_cmrx(sens_map)
        t,s,c,x,y = im_gnd_in.shape
        mask = cartesian_mask((t,s,x,y), acc = self.Acc) 
        mask_multi = np.tile(mask[:,:,np.newaxis, ...], (1, 1, 10, 1, 1))
        
        # under sample
        # cropped_k = fft2c_tensor(im_gnd_in)
        # undersampled_k = cropped_k * mask_multi
        # undersampled_img = ifft2c_tensor(undersampled_k)
        
        
        # combine under-samped and full smapled separately
        print(sens_map.shape,'sens')
        weighted_images = im_gnd_in * sens_map.conj()
        combined_image = torch.sum(weighted_images, axis=2)           
        # sensitivity_sum = torch.sum(sens_map * sens_map.conj(), axis=1)
        # print(sensitivity_sum.shape,'senssumm',combined_image.shape)
        final_image = norm_Im(combined_image)
        # print(final_image.max(),'combiend max')
        final_image_sp = np.tile(final_image[:,:,np.newaxis, ...], (1, 1, 10, 1, 1))
        split_pred = torch.tensor(final_image_sp) * sens_map
        split_k = fft2c_tensor(split_pred)
        undersampled_k = split_k * mask_multi
        undersampled_img = ifft2c_tensor(undersampled_k)        

        weighted_images_und = undersampled_img * sens_map.conj()
        combined_image_und = torch.sum(weighted_images_und, axis=2)           
        final_image_und = combined_image_und      

        print(final_image.shape, final_image_und.shape,'shapeee')
        t,s,x,y = final_image.shape
        randseed = random.randint(0, y-self.patch)

        final_image = final_image[...,randseed:randseed+self.patch]
        final_image_und = final_image_und[...,randseed:randseed+self.patch]
     
        im_gnd_in = torch.from_numpy(to_tensor_format(final_image))
        im_und_in = torch.from_numpy(to_tensor_format(final_image_und))


        for i in range(s):
            im_gnd_out = im_gnd_in[:,:,i,...]
            im_und_out = im_und_in[:,:,i,...]
            yield im_und_out, im_gnd_out
        
    def __len__(self):
        return self.n_suj



class LoadTestDataMulti(data.Dataset):
    def __init__(self, Acc, ax):
        super(LoadTestDataMulti, self).__init__()
        self.data_gnd = '/bigdata/zhenlin/CMRxRecon2023Data/TestData/Multi_test/MultiCoil/Cine/TestSet/FullSample'
        self.sens_path = '/bigdata/zhenlin/CMRxRecon2023Data/TestData/Multi_test_Sens/MultiCoil/Cine/TestSet/FullSample'
        self.patient = [x for x in listdir(self.sens_path)]
        self.mask_path = "/bigdata/zhenlin/CMRxRecon2023Data/TestData/Single_test/Cine/TestSet/Cmask"
        self.sens_map = f'cine_{ax}_sen.mat'
        self.ax = 'cine_{}.mat'.format(ax)
        self.mask = 'mask_sax_ACS10.mat'.format(ax)
        del_list = []
        for i in range(len(self.patient)):
            if self.ax not in [x for x in listdir(join(self.data_gnd, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        self.patient = sorted(self.patient)
        print(len(self.patient), self.patient)
        self.n_suj = len(self.patient)
        self.Acc = Acc

    def __getitem__(self, index):

        dir_gnd = join(self.data_gnd, self.patient[index], self.ax)
        dir_mask = join(self.mask_path, self.patient[index], f"Acc{self.Acc}", self.mask)
        dir_sens_map = join(self.sens_path, self.patient[index], self.sens_map)
        
        mask = scipy.io.loadmat(dir_mask)['data']
        mask_multi = np.tile(mask[:,:,np.newaxis, ...], (1, 1, 10, 1, 1))
        
        sens_map = scipy.io.loadmat(dir_sens_map)['SNS_sz']
        sens_map = torch.from_numpy(np.transpose(sens_map, (3,2,1,0)))
        k_gnd = load_mat(dir_gnd)
        im_gnd_in = torch.from_numpy(ifft2c(k_gnd))
        im_gnd_in = crop_cmrx(im_gnd_in)
        # im_gnd_in = norm_Im(im_gnd_in)
        
        sens_map = crop_cmrx(sens_map)
        
        # under sample
        # cropped_k = fft2c_tensor(im_gnd_in)
        # undersampled_k = cropped_k * mask_multi
        # undersampled_img = ifft2c_tensor(undersampled_k)
        
        
        # combine under-samped and full smapled separately
        print(sens_map.shape,'sens')
        weighted_images = im_gnd_in * sens_map.conj()
        combined_image = torch.sum(weighted_images, axis=2)           
        # sensitivity_sum = torch.sum(sens_map * sens_map.conj(), axis=1)
        # print(sensitivity_sum.shape,'senssumm',combined_image.shape)
        final_image = norm_Im(combined_image) 
        final_image_sp = np.tile(final_image[:,:,np.newaxis, ...], (1, 1, 10, 1, 1))
        split_pred = torch.tensor(final_image_sp) * sens_map
        split_k = fft2c_tensor(split_pred)
        undersampled_k = split_k * mask_multi
        undersampled_img = ifft2c_tensor(undersampled_k)        

        weighted_images_und = undersampled_img * sens_map.conj()
        combined_image_und = torch.sum(weighted_images_und, axis=2)           
        final_image_und = combined_image_und 
        # / (sensitivity_sum + 1e-8)      

        print(final_image.shape,'shapeee')
        t,s,x,t = final_image.shape
     
        im_gnd_in = torch.from_numpy(to_tensor_format(final_image))
        im_und_in = torch.from_numpy(to_tensor_format(final_image_und))
        mask = np.tile(mask[:,np.newaxis, ...], (1, 2, 1, 1, 1))
        mask = torch.from_numpy(mask)

        for i in range(s):
            im_gnd_out = im_gnd_in[:,:,i,...]
            im_und_out = im_und_in[:,:,i,...]
            im_und_out2 = undersampled_img[:,i,...]
            mask_out = mask[:,:,i,...]
            sens_out = sens_map[i,...]
            # print(im_und_out.shape,im_gnd_out.shape)
            yield im_und_out, im_gnd_out, mask_out, sens_out, im_und_out2
        
    def __len__(self):
        return self.n_suj




