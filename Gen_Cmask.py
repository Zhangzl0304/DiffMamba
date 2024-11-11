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
import scipy.io

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

def to_pseudo_real(data):
    '''
    input: tensor of shape [t,1,x,y]
    output: tensor of shape [t,2,x,y]
    '''
    return torch.cat((data.real, data.imag), dim=1)

def to_complex(data):
    '''
    input: tensor of shape [t,2,x,y]
    output: tensor of shape [t,1,x,y]
    '''
    return data[:,0:1,...]+data[:,1:2,...]*1j

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
    x = x / np.max(x)
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




def cal_lines(org_mask):
    '''
    this function calaulate the total sampled lines in a mask
    '''
    lines = 0
    for i in org_mask:
        if i[0] != 0:
            lines += 1
    return lines


def k_space_avg(k_space_data, mask):
    '''
    this function average the under sampled k-space data according to its mask
    input: k space data with shape [t, 2, x, y], tensor
           mask [t, 2, x, y], tensor
    output: averaged k space data [t, 2, x, y], tensor
    
    '''
    org_lines = cal_lines(mask[0,0,...])
#     plt.imshow(mask[0,0,...], cmap='gray')
#     plt.show()   
#     print("org_lines",org_lines)
    avg_cons = torch.sum(mask, 0)
    lines = cal_lines(avg_cons[0,...])
#     print('avg_lines',lines)
#     print(avg_cons, avg_cons.shape)
    mask_sum = avg_cons[0,...]
    for lines in mask_sum:
#         print(lines)
        for j in range(len(lines)):
            if lines[j] == 0:
                lines[j] = 1
#     print(mask_sum)
    lines = cal_lines(mask_sum)
#     print(lines)
    k_avg = torch.div(torch.sum(k_space_data, 0), mask_sum.unsqueeze(0).unsqueeze(0))
    return k_avg


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



class TrainDataFolder(data.Dataset):
    def __init__(self, Acc, ax, tsfm, norm, val = False):
        super(TrainDataFolder, self).__init__()
        self.path = r'/rds/general/user/zz8523/home/CMRrecon/SingleCoil/Cine/TrainingSet/AccFactor{}'.format(Acc)
        self.patient = [x for x in listdir(self.path)]
        self.data_gnd = '/rds/general/user/zz8523/home/CMRrecon/SingleCoil/Cine/TrainingSet/FullSample'
        self.norm = norm_01_im
        self.lax = 'cine_{}.mat'.format(ax)
        self.mask = 'cine_{}_mask.mat'.format(ax)
        del_list = []
        for i in range(len(self.patient)):
            if self.lax not in [x for x in listdir(join(self.path, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        if not val:
            pass
            # self.patient.pop()
        else:
            self.patient = [self.patient.pop()]
        print(len(self.patient), self.patient)
        self.patient = sorted(self.patient)
        self.n_suj = len(self.patient)
        self.tsfm = tsfm
        self.norm = norm


    def __getitem__(self, index):

        dir_gnd = join(self.data_gnd, self.patient[index], self.lax)
        k_gnd = load_mat(dir_gnd)
        im_gnd = ifft2c(k_gnd)
        
        t,s,x,y = im_gnd.shape
        self.slicenum = s
        print(im_gnd.shape)

        im_gnd_in = torch.from_numpy(im_gnd)
        im_gnd_in = CenterCrop(im_gnd_in)
        
        cropped_k = fft2c(im_gnd_in)
        
        
        mask = cartesian_mask((t,s,128,128), 8)
        undersampled_k = torch.from_numpy(cropped_k) * mask
        im_und_in = torch.from_numpy(ifft2c(undersampled_k))
        mask = torch.from_numpy(mask)
        
        if self.tsfm:
           im_cat = torch.cat((im_gnd_in.unsqueeze(0), im_und_in.unsqueeze(0), mask.unsqueeze(0)), dim=0)
           im_cat_in = self.tsfm(im_cat)
           im_gnd_in = im_cat_in[0,...]
           im_und_in = im_cat_in[1,...]
           mask = im_cat_in[2,...]
        
        im_gnd_in_real = torch.cat((im_gnd_in.real.unsqueeze(0), im_gnd_in.imag.unsqueeze(0)), dim=0)
        im_gnd_in_real = Norm_01(im_gnd_in_real)
        im_gnd_in = im_gnd_in_real[0,...]+im_gnd_in_real[1,...]*1j
        
        im_und_in_real = torch.cat((im_und_in.real.unsqueeze(0), im_und_in.imag.unsqueeze(0)), dim=0)
        im_und_in_real = Norm_01(im_und_in_real)
        im_und_in = im_und_in_real[0,...]+im_und_in_real[1,...]*1j
        
        for i in range(s):
            im_gnd_out = im_gnd_in[:,i:i+1,...]
            im_und_out = im_und_in[:,i:i+1,...]
            mask_out1 = mask[:,i:i+1,...]
            k_avg = k_space_avg(fft2c_tensor(im_und_out), mask_out1)
            im_avg_und = ifft2c_tensor(k_avg)
            im_avg_gnd = torch.sum(im_gnd_out, 0).unsqueeze(0)
            im_avg_und = to_pseudo_real(im_avg_und).float()
            im_avg_gnd = to_pseudo_real(im_avg_gnd).float()
            
            yield im_avg_und, im_avg_gnd, mask_out1


    def __len__(self):
        return self.n_suj
    
    
class TrainDataFolderRes(data.Dataset):
    def __init__(self, Acc, Accmask, ACS,ax, tsfm, norm, val = False):
        super(TrainDataFolderRes, self).__init__()
        self.path = r'/rds/general/user/zz8523/home/CMRrecon/SingleCoil/Cine/TrainingSet/AccFactor{}'.format(Acc)
        self.patient = [x for x in listdir(self.path)]
        self.data_gnd = '/rds/general/user/zz8523/home/CMRrecon/SingleCoil/Cine/TrainingSet/FullSample'
#         self.norm = norm_01_im
        self.lax = 'cine_{}.mat'.format(ax)
        self.mask = 'cine_{}_mask.mat'.format(ax)
        del_list = []
        for i in range(len(self.patient)):
            # print(self.patient[i],i)
            # print([x for x in listdir(join(self.path, self.patient[i]))], i)
            if self.lax not in [x for x in listdir(join(self.path, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        if not val:
            pass
        else:
            self.patient = [self.patient.pop()]
        print(len(self.patient), self.patient)
        self.patient = sorted(self.patient)
        self.n_suj = len(self.patient)
        self.tsfm = tsfm
        self.norm = norm
        self.Accmask = Accmask
        self.ACS = ACS

    def __getitem__(self, index):
        
        dir_gnd = join(self.data_gnd, self.patient[index], self.lax)
        k_gnd = load_mat(dir_gnd)
        im_gnd = ifft2c(k_gnd)
        
        t,s,x,y = im_gnd.shape
        self.slicenum = s
        print(im_gnd.shape)

        im_gnd_in = torch.from_numpy(im_gnd)
        im_gnd_in = CenterCrop(im_gnd_in)
        cropped_k = fft2c(im_gnd_in)
        
        
        mask = cartesian_mask((t,s,128,128), self.Accmask, sample_n=self.ACS)
        undersampled_k = torch.from_numpy(cropped_k) * mask
        im_und_in = torch.from_numpy(ifft2c(undersampled_k))
        
        mask = torch.from_numpy(mask)
        
        if self.tsfm:
           im_cat = torch.cat((im_gnd_in.unsqueeze(0), im_und_in.unsqueeze(0), mask.unsqueeze(0)), dim=0)
           im_cat_in = self.tsfm(im_cat)
           im_gnd_in = im_cat_in[0,...]
           im_und_in = im_cat_in[1,...]
           mask = im_cat_in[2,...]
        
        im_gnd_in_real = torch.cat((im_gnd_in.real.unsqueeze(0), im_gnd_in.imag.unsqueeze(0)), dim=0)
        im_gnd_in_real = Norm_01(im_gnd_in_real)
        im_gnd_in = im_gnd_in_real[0,...]+im_gnd_in_real[1,...]*1j
        
        im_und_in_real = torch.cat((im_und_in.real.unsqueeze(0), im_und_in.imag.unsqueeze(0)), dim=0)
        im_und_in_real = Norm_01(im_und_in_real)
        im_und_in = im_und_in_real[0,...]+im_und_in_real[1,...]*1j
        
        for i in range(s):
            im_gnd_out = im_gnd_in[:,i:i+1,...]
            im_und_out = im_und_in[:,i:i+1,...]
            mask_out1 = mask[:,i:i+1,...]
            k_avg = k_space_avg(fft2c_tensor(im_und_out), mask_out1)
            im_avg_und = ifft2c_tensor(k_avg)
            im_res_und = im_und_out - im_avg_und
            im_avg_gnd = torch.sum(im_gnd_out, 0).unsqueeze(0)
            im_res_gnd = im_gnd_out - im_avg_gnd
            im_res_und = to_pseudo_real(im_res_und).float()
            im_res_gnd = to_pseudo_real(im_res_gnd).float()        
            
            yield im_res_und, im_res_gnd, mask_out1


    def __len__(self):
        return self.n_suj


class TestDataFolderRes(data.Dataset):
    def __init__(self, Acc, Accmask, ACS,ax, norm, val = False):
        super(TestDataFolderRes, self).__init__()
        self.path = r'/rds/general/user/zz8523/home/test/Cine/TestSet/AccFactor{}'.format(Acc)
        self.patient = [x for x in listdir(self.path)]
        self.data_gnd = '/rds/general/user/zz8523/home/test/Cine/TestSet/FullSample'
#         self.norm = norm_01_im
        self.lax = 'cine_{}.mat'.format(ax)
        self.mask = 'cine_{}_mask.mat'.format(ax)
        del_list = []
        for i in range(len(self.patient)):
            # print(self.patient[i],i)
            # print([x for x in listdir(join(self.path, self.patient[i]))], i)
            if self.lax not in [x for x in listdir(join(self.path, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        if not val:
            pass
        else:
            self.patient = [self.patient.pop()]
        print(len(self.patient), self.patient)
        self.patient = sorted(self.patient)
        self.n_suj = len(self.patient)
        self.norm = norm
        self.Accmask = Accmask
        self.ACS = ACS

    def __getitem__(self, index):
        
        dir_gnd = join(self.data_gnd, self.patient[index], self.lax)
        k_gnd = load_mat(dir_gnd)
        im_gnd = ifft2c(k_gnd)
        
        t,s,x,y = im_gnd.shape
        self.slicenum = s
        print(im_gnd.shape)

        im_gnd_in = torch.from_numpy(im_gnd)
        im_gnd_in = CenterCrop(im_gnd_in)
        cropped_k = fft2c(im_gnd_in)
        
        
        mask = cartesian_mask((t,s,128,128), self.Accmask, sample_n=self.ACS)
        undersampled_k = torch.from_numpy(cropped_k) * mask
        im_und_in = torch.from_numpy(ifft2c(undersampled_k))
        mask = torch.from_numpy(mask)
        
        im_gnd_in_real = torch.cat((im_gnd_in.real.unsqueeze(0), im_gnd_in.imag.unsqueeze(0)), dim=0)
        im_gnd_in_real = Norm_01(im_gnd_in_real)
        im_gnd_in = im_gnd_in_real[0,...]+im_gnd_in_real[1,...]*1j
        
        im_und_in_real = torch.cat((im_und_in.real.unsqueeze(0), im_und_in.imag.unsqueeze(0)), dim=0)
        im_und_in_real = Norm_01(im_und_in_real)
        im_und_in = im_und_in_real[0,...]+im_und_in_real[1,...]*1j
        
        for i in range(s):
            im_gnd_out = im_gnd_in[:,i:i+1,...]
            im_und_out = im_und_in[:,i:i+1,...]
            mask_out1 = mask[:,i:i+1,...]
            k_avg = k_space_avg(fft2c_tensor(im_und_out), mask_out1)
            im_avg_und = ifft2c_tensor(k_avg)
            im_res_und = im_und_out - im_avg_und
            im_avg_gnd = torch.sum(im_gnd_out, 0).unsqueeze(0)
            im_res_gnd = im_gnd_out - im_avg_gnd
            im_res_und = to_pseudo_real(im_res_und).float()
            im_res_gnd = to_pseudo_real(im_res_gnd).float()        
            
            yield im_res_und, im_res_gnd, mask_out1


    def __len__(self):
        return self.n_suj


from improved_diffusion.Evaluation import ssim, mse, nmse, psnr

class DataBaseRes(data.Dataset):
    def __init__(self, Acc, Accmask, ACS,ax, norm, val = False):
        super(TestDataFolderRes, self).__init__()
        self.path = r'/rds/general/user/zz8523/home/test/Cine/TestSet/AccFactor{}'.format(Acc)
        self.patient = [x for x in listdir(self.path)]
        self.data_gnd = '/rds/general/user/zz8523/home/test/Cine/TestSet/FullSample'
#         self.norm = norm_01_im
        self.lax = 'cine_{}.mat'.format(ax)
        self.mask = 'cine_{}_mask.mat'.format(ax)
        del_list = []
        for i in range(len(self.patient)):
            # print(self.patient[i],i)
            # print([x for x in listdir(join(self.path, self.patient[i]))], i)
            if self.lax not in [x for x in listdir(join(self.path, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        if not val:
            pass
        else:
            self.patient = [self.patient.pop()]
        print(len(self.patient), self.patient)
        self.patient = sorted(self.patient)
        self.n_suj = len(self.patient)
        self.norm = norm
        self.Accmask = Accmask
        self.ACS = ACS

    def __getitem__(self, index):
        
        dir_gnd = join(self.data_gnd, self.patient[index], self.lax)
        k_gnd = load_mat(dir_gnd)
        im_gnd = ifft2c(k_gnd)
        
        t,s,x,y = im_gnd.shape
        self.slicenum = s
        print(im_gnd.shape)

        im_gnd_in = torch.from_numpy(im_gnd)
        im_gnd_in = CenterCrop(im_gnd_in)
        cropped_k = fft2c(im_gnd_in)
        
        
        mask = cartesian_mask((t,s,128,128), self.Accmask, sample_n=self.ACS)
        undersampled_k = torch.from_numpy(cropped_k) * mask
        im_und_in = torch.from_numpy(ifft2c(undersampled_k))
        mask = torch.from_numpy(mask)
        
        im_gnd_in_real = torch.cat((im_gnd_in.real.unsqueeze(0), im_gnd_in.imag.unsqueeze(0)), dim=0)
        im_gnd_in_real = Norm_01(im_gnd_in_real)
        im_gnd_in = im_gnd_in_real[0,...]+im_gnd_in_real[1,...]*1j
        
        im_und_in_real = torch.cat((im_und_in.real.unsqueeze(0), im_und_in.imag.unsqueeze(0)), dim=0)
        im_und_in_real = Norm_01(im_und_in_real)
        im_und_in = im_und_in_real[0,...]+im_und_in_real[1,...]*1j
        
        for i in range(s):
            im_gnd_out = im_gnd_in[:,i:i+1,...]
            im_und_out = im_und_in[:,i:i+1,...]
            mask_out1 = mask[:,i:i+1,...]
            k_avg = k_space_avg(fft2c_tensor(im_und_out), mask_out1)
            im_avg_und = ifft2c_tensor(k_avg)
            im_res_und = im_und_out - im_avg_und
            im_avg_gnd = torch.sum(im_gnd_out, 0).unsqueeze(0)
            im_res_gnd = im_gnd_out - im_avg_gnd
            im_res_und = to_pseudo_real(im_res_und).float()
            im_res_gnd = to_pseudo_real(im_res_gnd).float()        
            
            yield im_res_und, im_res_gnd, mask_out1


    def __len__(self):
        return self.n_suj
    
class TestDataFolder(data.Dataset):
    def __init__(self, Acc, ax, norm):
        super(TestDataFolder, self).__init__()
        self.path = r'/rds/general/user/zz8523/home/test/Cine/TestSet/AccFactor{}'.format(Acc)
        self.patient = [x for x in listdir(self.path)]
        self.data_gnd = '/rds/general/user/zz8523/home/test/Cine/TestSet/FullSample'
        self.norm = norm_01_im
        # self.patch = 36
        # self.patch_num = 3
        self.lax = 'cine_{}.mat'.format(ax)
        self.mask = 'cine_{}_mask.mat'.format(ax)
        del_list = []
        for i in range(len(self.patient)):
            # print(self.patient[i],i)
            # print([x for x in listdir(join(self.path, self.patient[i]))], i)
            if self.lax not in [x for x in listdir(join(self.path, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        self.patient = sorted(self.patient)
        print(len(self.patient), self.patient)
        self.n_suj = len(self.patient)


    def __getitem__(self, index):

        dir_gnd = join(self.data_gnd, self.patient[index], self.lax)
        k_gnd = load_mat(dir_gnd)
        im_gnd = ifft2c(k_gnd)
        
        t,s,x,y = im_gnd.shape
        self.slicenum = s
        print(im_gnd.shape)

        im_gnd_in = torch.from_numpy(im_gnd)
        im_gnd_in = CenterCrop(im_gnd_in)
        cropped_k = fft2c(im_gnd_in)
        
        
        mask = cartesian_mask((t,s,128,128), 2.56)
        undersampled_k = torch.from_numpy(cropped_k) * mask
        im_und_in = torch.from_numpy(ifft2c(undersampled_k))
        mask = torch.from_numpy(mask)
        print(im_gnd_in.shape, im_und_in.shape,'sssss')
        im_gnd_in_real = torch.cat((im_gnd_in.real.unsqueeze(0), im_gnd_in.imag.unsqueeze(0)), dim=0)
        im_gnd_in_real = Norm_01(im_gnd_in_real)
        im_gnd_in = im_gnd_in_real[0,...]+im_gnd_in_real[1,...]*1j
        
        im_und_in_real = torch.cat((im_und_in.real.unsqueeze(0), im_und_in.imag.unsqueeze(0)), dim=0)
        im_und_in_real = Norm_01(im_und_in_real)
        im_und_in = im_und_in_real[0,...]+im_und_in_real[1,...]*1j
        print(im_gnd_in.shape, im_und_in.shape,'sssssafter')

        for i in range(s):
            im_gnd_out = im_gnd_in[:,i:i+1,...]
            im_und_out = im_und_in[:,i:i+1,...]
            mask_out1 = mask[:,i:i+1,...]
            k_avg = k_space_avg(fft2c_tensor(im_und_out), mask_out1)
            im_avg_und = ifft2c_tensor(k_avg)
            im_avg_gnd = torch.sum(im_gnd_out, 0).unsqueeze(0)
            im_avg_und = to_pseudo_real(im_avg_und).float()
            im_avg_gnd = to_pseudo_real(im_avg_gnd).float()          
            
            yield im_avg_und, im_avg_gnd, mask_out1
    def __len__(self):
        return self.n_suj

class TestDataFolder2(data.Dataset):
    def __init__(self, Acc, ax, norm):
        super(TestDataFolder2, self).__init__()
        self.path = r'/rds/general/user/zz8523/home/test/Cine/TestSet/AccFactor{}'.format(Acc)
        self.patient = [x for x in listdir(self.path)]
        self.data_gnd = '/rds/general/user/zz8523/home/test/Cine/TestSet/FullSample'
        self.norm = norm_01_im
        self.lax = 'cine_{}.mat'.format(ax)
        self.mask = 'cine_{}_mask.mat'.format(ax)
        del_list = []
        for i in range(len(self.patient)):
            # print(self.patient[i],i)
            # print([x for x in listdir(join(self.path, self.patient[i]))], i)
            if self.lax not in [x for x in listdir(join(self.path, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        self.patient = sorted(self.patient)
        print(len(self.patient), self.patient)
        self.n_suj = len(self.patient)


    def __getitem__(self, index):

        dir_gnd = join(self.data_gnd, self.patient[index], self.lax)
        k_gnd = load_mat(dir_gnd)
        im_gnd = ifft2c(k_gnd)
        
        t,s,x,y = im_gnd.shape
        self.slicenum = s
        print(im_gnd.shape)

        im_gnd_in = torch.from_numpy(im_gnd)
        im_gnd_in = CenterCrop(im_gnd_in)
        cropped_k = fft2c(im_gnd_in)
        
        
        mask = cartesian_mask((t,s,128,128), 2.56)
        undersampled_k = torch.from_numpy(cropped_k) * mask
        im_und_in = torch.from_numpy(ifft2c(undersampled_k))
        mask = torch.from_numpy(mask)

        im_gnd_in_real = torch.cat((im_gnd_in.real.unsqueeze(0), im_gnd_in.imag.unsqueeze(0)), dim=0)
        im_gnd_in_real = Norm_01(im_gnd_in_real)
        im_gnd_in = im_gnd_in_real[0,...]+im_gnd_in_real[1,...]*1j
        
        im_und_in_real = torch.cat((im_und_in.real.unsqueeze(0), im_und_in.imag.unsqueeze(0)), dim=0)
        im_und_in_real = Norm_01(im_und_in_real)
        im_und_in = im_und_in_real[0,...]+im_und_in_real[1,...]*1j
        
        for i in range(s):
            im_gnd_out = im_gnd_in[:,i:i+1,...]
            im_und_out = im_und_in[:,i:i+1,...]
            mask_out1 = mask[:,i:i+1,...]
            k_avg = k_space_avg(fft2c_tensor(im_und_out), mask_out1)
            im_avg_und = ifft2c_tensor(k_avg)
            im_avg_gnd = torch.sum(im_gnd_out, 0).unsqueeze(0)
            im_avg_und = to_pseudo_real(im_avg_und).float()
            im_avg_gnd = to_pseudo_real(im_avg_gnd).float()   
            
            
            yield im_avg_und, im_avg_gnd, mask_out1
    def __len__(self):
        return self.n_suj

import os

class Genmask(data.Dataset):
    def __init__(self, Acc, ACS, ax):
        super(Genmask, self).__init__()
        # self.path = r'/rds/general/user/zz8523/home/test/Cine/TestSet/AccFactor04'.format(Acc)
        # self.patient = [x for x in listdir(self.path)]
        self.data_gnd = '/bigdata/zhenlin/CMRxRecon2023Data/TestData/Multi_test/MultiCoil/Cine/TestSet/FullSample'
        self.patient = [x for x in listdir(self.data_gnd)]
        self.cmask_path = '/bigdata/zhenlin/CMRxRecon2023Data/TestData/Cmask'
        self.lax = 'cine_{}.mat'.format(ax)
        self.mask = 'cine_{}_mask.mat'.format(ax)
        del_list = []
        for i in range(len(self.patient)):
            if self.lax not in [x for x in listdir(join(self.data_gnd, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        self.patient = sorted(self.patient)
        print(len(self.patient), self.patient)
        self.n_suj = len(self.patient)
        self.acc = Acc
        self.ACS = ACS

    def __getitem__(self, index):

        dir_gnd = join(self.data_gnd, self.patient[index], self.lax)
        k_gnd = load_mat(dir_gnd)
        im_gnd = ifft2c(k_gnd)
        im_gnd = crop_cmrx(im_gnd)
        
        t,s,c,x,y = im_gnd.shape
        print('im_gnd.shape',im_gnd.shape)
        mask = cartesian_mask((t,s,x,y), self.acc, self.ACS)
        save_path = join(self.cmask_path, f"Acc{self.acc}_ACS{self.ACS}", self.patient[index])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        scipy.io.savemat(join(save_path,'mask_sax_ACS10.mat'), {'data': mask})
        print(f"save {self.patient[index]} mask finished, shape {mask.shape}")
        

    def __len__(self):
        return self.n_suj


# mask_gene = Genmask(4, 10, 'sax')
# for i in range(len(mask_gene)):
#     print(i)
#     mask_gene[i]



class GenMultiData(data.Dataset):
    def __init__(self, Acc, ax):
        super(GenMultiData, self).__init__()
        self.data_gnd = '/bigdata/zhenlin/CMRxRecon2023Data/ChallengeData/MultiCoil/Cine/TrainingSet/FullSample'
        self.patient = [x for x in listdir(self.data_gnd)]
        self.ax_file = 'cine_{}.mat'.format(ax)
        self.sens_map = f'cine_{ax}_sen.mat'
        self.sens_path = '/bigdata/zhenlin/CMRxRecon2023Data/ChallengeData/Sens/MultiCoil/Cine/TrainingSet/FullSample'
        del_list = []
        for i in range(len(self.patient)):
            if self.ax_file not in [x for x in listdir(join(self.data_gnd, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        random.shuffle(self.patient)
        print(len(self.patient), self.patient)
        self.n_suj = len(self.patient)
        self.acc = Acc
        self.ax = ax
 


    def __getitem__(self, index):

        dir_gnd = join(self.data_gnd, self.patient[index], self.ax_file)
        dir_sens_map = join(self.sens_path, self.patient[index], self.sens_map)

        k_gnd = load_mat(dir_gnd)
        sens_map = scipy.io.loadmat(dir_sens_map)['SNS_sz']
        sens_map = np.transpose(sens_map, (3,2,1,0))
        
        im_gnd_in = ifft2c(k_gnd)

        im_gnd_in = crop_cmrx(im_gnd_in)
        sens_map = crop_cmrx(sens_map)
        im_gnd_in = norm_Im(im_gnd_in)
        print(im_gnd_in.shape)
        # print('single cons ful',np.max(im_gnd_in))
        # print('mean',np.mean(im_gnd_in),np.max(im_gnd_in),np.min(im_gnd_in))
        t,s,c,x,y = im_gnd_in.shape
        mask = cartesian_mask((t,s,x,y), acc = self.acc)        
        mask_multi = np.tile(mask[:,:,np.newaxis, ...], (1, 1, 10, 1, 1))
        cropped_k = fft2c(im_gnd_in)
        undersampled_k = cropped_k * mask_multi
        undersampled_img = ifft2c(undersampled_k)
        # print('single cons und',np.max(undersampled_img))
        
        weighted_images = im_gnd_in * sens_map.conj()
        combined_image = np.sum(weighted_images, axis=2)           
        sensitivity_sum = np.sum(sens_map * sens_map.conj(), axis=1)
        final_image = combined_image / (sensitivity_sum + 1e-8)
        # plt.imshow(abs(final_image[0,0,...]), cmap = 'gray')
        # plt.show()
        # print('multi cons full',np.max(final_image))
        weighted_images_und = undersampled_img * sens_map.conj()
        combined_image_und = np.sum(weighted_images_und, axis=2)           
        # print(np.mean(sensitivity_sum))
        final_image_und = combined_image_und / (sensitivity_sum + 1e-8)        
        # plt.imshow(abs(final_image_und[0,0,...]), cmap = 'gray')
        # plt.show()
        # print('multi cons und',np.max(final_image_und))
        psnrone = psnr(abs(final_image)/abs(final_image).max(), abs(final_image_und) / abs(final_image_und).max())
        print('psnr',psnrone)
        
        save_dir = join(self.sens_path, self.patient[index])
        save_dir = join(save_dir, f'data_{self.ax}_combined.npz')
        
        np.savez(save_dir, data_gnd=final_image, data_und=final_image_und, mask = mask)
        print(f'data for {self.patient[index]} saved at {save_dir}! ')
        # final_k = fft2c(final_image)
        # final_k_und = mask*final_k
        # final_conbined_im = ifft2c(final_k_und)
        # plt.imshow(abs(final_conbined_im[0,0,...]), cmap = 'gray')
        # plt.show()       
        # psnrone = psnr(abs(final_image)/abs(final_image).max(), abs(final_conbined_im) / abs(final_conbined_im).max())
        # print('psnr_conbminfirst',psnrone)        
        
    def __len__(self):
        return self.n_suj


class GenMultiTestData(data.Dataset):
    def __init__(self, Acc, ax):
        super(GenMultiTestData, self).__init__()
        self.data_gnd = '/bigdata/zhenlin/CMRxRecon2023Data/TestData/Multi_test/MultiCoil/Cine/TestSet/FullSample'
        self.patient = [x for x in listdir(self.data_gnd)]
        self.mask_path = f"/bigdata/zhenlin/CMRxRecon2023Data/TestData/Cmask/Acc{Acc}_ACS10"
        self.sens_map = f'cine_{ax}_sen.mat'
        self.sens_path = '/bigdata/zhenlin/CMRxRecon2023Data/TestData/Multi_test_Sens/MultiCoil/Cine/TestSet/FullSample'
        self.ax = ax
        self.mask = 'mask_sax_ACS10.mat'.format(ax)
        self.ax_file = 'cine_{}.mat'.format(ax)
        del_list = []
        for i in range(len(self.patient)):
            if self.ax_file not in [x for x in listdir(join(self.data_gnd, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        self.patient = sorted(self.patient)
        print(len(self.patient), self.patient)
        self.n_suj = len(self.patient)
        self.Acc = Acc

    def __getitem__(self, index):

        dir_gnd = join(self.data_gnd, self.patient[index], self.ax_file)
        dir_sens_map = join(self.sens_path, self.patient[index], self.sens_map)
        dir_mask = join(self.mask_path, self.patient[index],self.mask)
        mask = scipy.io.loadmat(dir_mask)['data']
        
        k_gnd = load_mat(dir_gnd)
        sens_map = scipy.io.loadmat(dir_sens_map)['SNS_sz']
        sens_map = np.transpose(sens_map, (3,2,1,0))
        
        im_gnd_in = ifft2c(k_gnd)

        im_gnd_in = crop_cmrx(im_gnd_in)
        sens_map = crop_cmrx(sens_map)
        im_gnd_in = norm_Im(im_gnd_in)
        print(im_gnd_in.shape)

        t,s,c,x,y = im_gnd_in.shape
        # mask = cartesian_mask((t,s,x,y), acc = self.acc)        
        mask_multi = np.tile(mask[:,:,np.newaxis, ...], (1, 1, 10, 1, 1))
        cropped_k = fft2c(im_gnd_in)
        undersampled_k = cropped_k * mask_multi
        undersampled_img = ifft2c(undersampled_k)
        # print('single cons und',np.max(undersampled_img))
        
        weighted_images = im_gnd_in * sens_map.conj()
        combined_image = np.sum(weighted_images, axis=2)           
        sensitivity_sum = np.sum(sens_map * sens_map.conj(), axis=1)
        final_image = combined_image 
        # / (sensitivity_sum + 1e-8)
        # plt.imshow(abs(final_image[0,0,...]), cmap = 'gray')
        # plt.show()
        # print('multi cons full',np.max(final_image))
        weighted_images_und = undersampled_img * sens_map.conj()
        combined_image_und = np.sum(weighted_images_und, axis=2)           
        # print(np.mean(sensitivity_sum))
        final_image_und = combined_image_und 
        # / (sensitivity_sum + 1e-8)        
        # plt.imshow(abs(final_image_und[0,0,...]), cmap = 'gray')
        # plt.show()
        # print('multi cons und',np.max(final_image_und))
        psnrone = psnr(abs(final_image)/abs(final_image).max(), abs(final_image_und) / abs(final_image_und).max())
        print('psnr',psnrone)
        
        save_dir = join(self.mask_path, self.patient[index])
        save_dir = join(save_dir, f'data_{self.ax}_combined.npz')
        
        np.savez(save_dir, data_gnd=final_image, data_und=final_image_und)
        print(f'data for {self.patient[index]} saved at {save_dir}! ')
        
    def __len__(self):
        return self.n_suj

Generateion = GenMultiData(4, 'sax')
for i in range(len(Generateion)):
    Generateion[i]

# Generateion = GenMultiData(4, 'sax')
# for i in range(len(Generateion)):
#     Generateion[i]