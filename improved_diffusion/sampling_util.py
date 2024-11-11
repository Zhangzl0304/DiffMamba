import math
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.utils as tvu
from PIL import Image
from sklearn.metrics import f1_score, jaccard_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from . import dist_util
from .metrics import FBound_metric, WCov_metric
# from datasets.monu import MonuDataset
from .Evaluation import ssim, mse, nmse, psnr
from dnn_io import from_tensor_format
import wandb
from data_loader_Cmask import ifft2c, fft2c

cityspallete = [
    0, 0, 0,
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    102, 102, 156,
    190, 153, 153,
    153, 153, 153,
    250, 170, 30,
    220, 220, 0,
    107, 142, 35,
    152, 251, 152,
    0, 130, 180,
    220, 20, 60,
    255, 0, 0,
    0, 0, 142,
    0, 0, 70,
    0, 60, 100,
    0, 80, 100,
    0, 0, 230,
    119, 11, 32,
]


result_save_path = "/home/zhenlin/DiffMamba/ISMRM/SSM_results/"
os.makedirs(result_save_path, exist_ok=True)


def calculate_metrics(x, gt):
    predict = x.detach().cpu().numpy().astype('uint8')
    target = gt.detach().cpu().numpy().astype('uint8')
    return f1_score(target.flatten(), predict.flatten()), jaccard_score(target.flatten(), predict.flatten()), \
           WCov_metric(predict, target), FBound_metric(predict, target)

def data_consistency(im_pred, im_und, mask):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    
    """
    k_pred = fft2c(im_pred)
    k_und = fft2c(im_und)
    # k_und = k_und/abs(k_und).max()
    # k_pred = k_pred/abs(k_pred).max()
    # print('k scale pred',abs(k_pred).max(), abs(k_pred).min(), abs(k_pred).mean())
    # print('k scale und',abs(k_und).max(), abs(k_und).min(),abs(k_und).mean())
    out = ifft2c((1 - mask) * k_pred + mask * k_und)
    return out
  

def data_consistency_Sens(im_pred, im_und, mask, sens_map):
    """
    impred    - input in k-space, should be [t, s, h, w]
    imund   - initially sampled elements in k-space, should be same as k
    mask - corresponding nonzero location [t, s, h, w]
    sems map: [s, coil, h, w]
    input all np.array
    retutn combined dc image
    """
    # assert im_pred.shape == im_und.shape
    
    sensitivity_sum = np.sum(sens_map * sens_map.conj(), axis=1)
    sensitivity_sum = np.tile(sensitivity_sum[:,np.newaxis, ...], (1, 10, 1, 1))
    # print(sensitivity_sum,'sens sum')
    im_pred = np.tile(im_pred[:,:,np.newaxis, ...], (1, 1, 10, 1, 1))
    im_und = np.tile(im_und[:,:,np.newaxis, ...], (1, 1, 10, 1, 1))
    print(im_pred.shape, im_und.shape, 'dcshape',sens_map.shape)
    split_pred = im_pred * sens_map * (sensitivity_sum + 1e-8)
    
    split_und = im_und * sens_map * (sensitivity_sum + 1e-8)
    
    k_pred = fft2c(split_pred)
    k_und = fft2c(split_und)
    
    mask = np.tile(mask[:,:,np.newaxis, ...], (1, 1, 10, 1, 1))
    out = ifft2c((1 - mask) * k_pred + mask * k_und)
    weighted_images = out * sens_map.conj()
    combined_image = np.sum(weighted_images, axis=2)  
    final_image = combined_image / (sensitivity_sum[:,0,...] + 1e-8)
    
    return final_image


def data_consistency_Sens_slice(im_pred, im_und, mask, sens_map):
    """
    im_pred    - input in k-space, should be [t, 1, h, w]
    im_und   - initially sampled elements in k-space, this time [t,coil,h,w]
    mask - corresponding nonzero location [t, 1, h, w]
    sems map: [1, coil, h, w]
    input all np.array
    retutn combined dc image [t,1,h,w]
    """
    # assert im_pred.shape == im_und.shape
    # sensitivity_sum = np.sum(sens_map * sens_map.conj(), axis=1)
    # sensitivity_sum = np.tile(sensitivity_sum[:,np.newaxis, ...], (1, 10, 1, 1))
    # print(sensitivity_sum,'sens sum')
    im_pred = np.tile(im_pred[:,:,np.newaxis, ...], (1, 1, 10, 1, 1))
    im_und = np.tile(im_und[:,np.newaxis, ...], (1, 1, 1, 1, 1))
    # print(im_pred.shape, im_und.shape, 'dcshape', sens_map.shape, mask.shape)
    split_pred = im_pred * sens_map 
    
    split_und = im_und 
    
    k_pred = fft2c(split_pred)
    k_und = fft2c(split_und)
    mask = np.tile(mask[:,:,np.newaxis, ...], (1, 1, 10, 1, 1))
    mask = mask[:,0:1,...]
    # print(mask.shape, 'mask shape')
    out = ifft2c((1 - mask) * k_pred + mask * k_und)

    weighted_images = out * sens_map.conj()
    combined_image = np.sum(weighted_images, axis=2)  
    final_image = combined_image 
    
    return final_image

def data_consistency_Sens_slicenorm(im_pred, im_und, mask, sens_map):
    """
    im_pred    - input in k-space, should be [t, 1, h, w]
    im_und   - initially sampled elements in k-space, should be same as k
    mask - corresponding nonzero location [t, 1, h, w]
    sems map: [1, coil, h, w]
    input all np.array
    retutn combined dc image [t,1,h,w]
    """
    assert im_pred.shape == im_und.shape
    # sensitivity_sum = np.sum(sens_map * sens_map.conj(), axis=1)
    # sensitivity_sum = np.tile(sensitivity_sum[:,np.newaxis, ...], (1, 10, 1, 1))
    # print(sensitivity_sum.shape,'sens sum')
    im_pred = np.tile(im_pred[:,:,np.newaxis, ...], (1, 1, 10, 1, 1))
    im_und = np.tile(im_und[:,:,np.newaxis, ...], (1, 1, 10, 1, 1))
    print(im_pred.shape, im_und.shape, 'dcshape',sens_map.shape)
    # sens_map = np.flip(sens_map, axis=0)
    split_pred = im_pred *sens_map
    split_und = im_und *sens_map
    
    k_pred = fft2c(split_pred)
    k_und = fft2c(split_und)
    mask = np.tile(mask[:,:,np.newaxis, ...], (1, 1, 10, 1, 1))
    mask = mask[:,0:1,...]
    # print(mask.shape, 'mask shape')
    out = ifft2c((1 - mask) * k_pred + mask * k_und)
    # out = ifft2c((1 - mask) * k_und + mask * k_und)
    # print(out.shape,'dcout')
    weighted_images = out * sens_map.conj()
    combined_image = np.sum(weighted_images, axis=2)  
    final_image = combined_image  
    # / (sensitivity_sum[:,0,...] + 1e-8)
    
    return final_image

def coil_combination_rss(coil_list):
    """
    :param coil_list: list of numpy arrays with shape [t, s, x, y]
    :return: numpy array with shape [t, s, x, y]
    """
    # 将列表中的数据堆叠成一个新的维度
    coil_data = np.stack(coil_list, axis=2)
    
    # 计算RSS组合
    rss_combined_image = np.sqrt(np.sum(np.abs(coil_data)**2, axis=2))
    
    return rss_combined_image

def cal_four_metrics(gt, target):
    '''
     both input have shape of [t,s,h,w]   
    '''

    psnr_value = psnr(np.array(abs(gt))/np.array(abs(gt)).max(), np.array(abs(target)) / np.array(abs(target)).max())
    ssim_value = ssim(np.array(abs(gt))/np.array(abs(gt)).max(), np.array(abs(target)) / np.array(abs(target)).max())
    nmse_value = nmse(np.array(abs(gt))/np.array(abs(gt)).max(), np.array(abs(target)) / np.array(abs(target)).max())
    mse_value = mse(np.array(abs(gt))/np.array(abs(gt)).max(), np.array(abs(target)) / np.array(abs(target)).max())
    return psnr_value, ssim_value, nmse_value, mse_value


def CMR_sampling_major_vote_func(diffusion_model, ddp_model, output_folder, dataset, logger, clip_denoised):
    ddp_model.eval()
    os.makedirs(output_folder, exist_ok=True)
    n_rounds = 1
    print(n_rounds,'nnnn')
    ssim_list, mse_list, nmse_list, psnr_list = [],[],[],[]

    with torch.no_grad():
      for data_idx in range(len(dataset)):
          one_sample = []
          gt_sample = []
          for slices in dataset[data_idx]:
              
            im_und, im_gnd, mask = slices
            # im_gnd = torch.squeeze(im_gnd, dim=0).float()
            # im_und = torch.squeeze(im_und, dim=0).float()
            print(im_gnd.shape, im_und.shape)
            gt_mask = im_gnd
            condition_on = im_und
            former_frame_for_feature_extraction = condition_on.to(dist_util.dev())

            x_pred = 0
            for round_index in range(n_rounds):
              print(f"Current Round: {round_index+1} / Total Round: {n_rounds}")

              model_kwargs = {
                "conditioned_image": former_frame_for_feature_extraction}

              x = diffusion_model.ddim_sample_loop(
                ddp_model,
                (former_frame_for_feature_extraction.shape[0], gt_mask.shape[1], former_frame_for_feature_extraction.shape[2],
                    former_frame_for_feature_extraction.shape[3]),
                progress=True,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs
            )
              pred_cur = x.cpu()
              x_pred = x_pred + np.array(pred_cur)
              

            x_pred_avg = x_pred/n_rounds
            show_img = x_pred_avg[:,0,...]+1j*x_pred_avg[:,1,...]
            show_gt = gt_mask[:,0,...]+1j*gt_mask[:,1,...]
            psnr_slice = psnr(np.array(abs(show_gt))/np.array(abs(show_gt)).max(), np.array(abs(show_img)) / np.array(abs(show_img)).max())
            print('psnr one slice',psnr_slice)
            one_sample.append(x_pred_avg)
            gt_sample.append(gt_mask)
          one_sample1 = np.array(one_sample)
          one_sample1 = np.transpose(one_sample1, (1,2,0,3,4))  
          gt_sample1 = np.array(gt_sample)
          gt_sample1 = np.transpose(gt_sample1, (1,2,0,3,4))
          one_sample_complex = from_tensor_format(one_sample1)
          gt_sample_complex = from_tensor_format(gt_sample1)
          
          image = wandb.Image(abs(one_sample_complex[0,0,...]), caption="one_sample")
          wandb.log({"finalimage": image})
          print(one_sample_complex.shape, 'one_sample1', gt_sample_complex.shape, 'gt_sample1')
          psnr_ = psnr(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_complex)) / np.array(abs(one_sample_complex)).max())
          print(psnr_,'onepsnr')
          mse_ = mse(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_complex)) / np.array(abs(one_sample_complex)).max())
          nmse_ = nmse(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_complex)) / np.array(abs(one_sample_complex)).max())
          ssim_ = ssim(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_complex)) / np.array(abs(one_sample_complex)).max())
          print(ssim_,'onessim')
          psnr_list.append(psnr_)
          mse_list.append(mse_)
          nmse_list.append(nmse_)
          ssim_list.append(ssim_)
          logger.info(
              f"n round psnr {psnr_list[-1]}, ssim {ssim_list[-1]}, mse {mse_list[-1]}, nmse {nmse_list[-1]}")

    my_length = len(psnr_list)
    print('psnr_lenth', my_length)

    logger.info("measure total avg")
    logger.info(f"mean psnr {np.mean(psnr_list)}")
    logger.info(f"mean ssim {np.mean(ssim_list)}")
    logger.info(f"mean mse {np.mean(mse_list)}")
    logger.info(f"mean nmse {np.mean(nmse_list)}")



def CMR_sampling_major_vote_func_with_dc(diffusion_model, ddp_model, output_folder, dataset, logger, clip_denoised, pic_save_path, npz_save_path):
    if not os.path.exists(pic_save_path):
        os.makedirs(pic_save_path)
    ddp_model.eval()
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(npz_save_path, exist_ok=True)
    n_rounds = 1
    print(n_rounds,'nnnn')
    ssim_list, mse_list, nmse_list, psnr_list = [],[],[],[]
    ssim_list_adc, mse_list_adc, nmse_list_adc, psnr_list_adc = [],[],[],[]
    with torch.no_grad():
      with open('/home/zhenlin/norm_test.log', 'r') as file:
          file_content = file.readlines()
      data_idx_list = [i for i in range(len(dataset))]
      # data_idx_list = [0]
      for data_idx in data_idx_list[3:]:
          gnd_norm_para = torch.tensor(float(file_content[data_idx].split()[5]))
          und_norm_para = torch.tensor(float(file_content[data_idx].split()[11]))
          sample_index = file_content[data_idx].split()[1]
          one_sample = []
          gt_sample = []
          und_sample = []
          one_sample_afterdc = []
          for idx, slices in enumerate(dataset[data_idx]):
            im_und, im_gnd, mask = slices
            print(im_gnd.shape, im_und.shape)
            gt_mask = im_gnd
            condition_on = im_und
            former_frame_for_feature_extraction = condition_on.to(dist_util.dev())
            x_pred = 0
            pred_cur_afterdc = 0
            for round_index in range(n_rounds):
              print(f"Current Round: {round_index+1} / Total Round: {n_rounds}")
              model_kwargs = {
                "conditioned_image": former_frame_for_feature_extraction}
              x = diffusion_model.ddim_sample_loop(
                ddp_model,
                (former_frame_for_feature_extraction.shape[0], gt_mask.shape[1], former_frame_for_feature_extraction.shape[2],
                    former_frame_for_feature_extraction.shape[3]),
                progress=True,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs
            )
              pred_cur = x.cpu()
              
              im_gnd_show = im_gnd[:,0,...]+1j*im_gnd[:,1,...]
              plt.imsave(pic_save_path + f'im_gnd_sample{sample_index}_slice{idx}.png',abs(im_gnd_show[0,...]), cmap='gray')
              im_und_show = im_und[:,0,...]+1j*im_und[:,1,...]
              plt.imsave(pic_save_path + f'im_und_sample{sample_index}_slice{idx}.png',abs(im_und_show[0,...]), cmap='gray')

              
              show_img_before_dc = pred_cur[:,0,...]+1j*pred_cur[:,1,...]
              plt.imsave(pic_save_path + f'before_dc_sample{sample_index}_slice{idx}.png',abs(show_img_before_dc[0,...]), cmap='gray')
              psnr_be = psnr(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_before_dc)) / np.array(abs(show_img_before_dc)).max())
              ssim_be = ssim(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_before_dc)) / np.array(abs(show_img_before_dc)).max())
              print(f'slice {idx} before dc', psnr_be, 'psnr', ssim_be, 'ssim')
#               print(pred_cur.shape, im_und.shape, mask.shape)
              print('pred scale',abs(pred_cur).max(), abs(pred_cur).min(), abs(pred_cur).mean(),'normed',abs(pred_cur*gnd_norm_para).max())
              print('und scale',abs(im_und).max(), abs(im_und).min(), abs(im_und).mean(),'normed',abs(im_und*und_norm_para).max())
              # for factor in np.arange(0.60, 0.65, 0.002):
                
              pred_cur_afterdc = data_consistency(pred_cur, im_und, mask)
                #  print('gnd norm',gnd_norm_para, 'und norm',und_norm_para)
              show_img_after_dc = pred_cur_afterdc[:,0,...]+1j*pred_cur_afterdc[:,1,...]
              plt.imsave(pic_save_path + f'after_dc_sample{sample_index}_slice{idx}.png',abs(show_img_after_dc[0,...]), cmap='gray')
              
              psnr_af = psnr(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_after_dc)) / np.array(abs(show_img_after_dc)).max())
              ssim_af = ssim(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_after_dc)) / np.array(abs(show_img_after_dc)).max())
              print(f'slice {idx} after dc',psnr_af, 'psnr', ssim_af, 'ssim')
              
              error = abs(im_gnd_show[0,...])-abs(show_img_before_dc[0,...])
              plt.imsave(pic_save_path + f'error_sample{sample_index}_slice{idx}.png',error, cmap='gray')

              x_pred = x_pred + np.array(pred_cur)
              pred_cur_afterdc = pred_cur_afterdc + np.array(pred_cur_afterdc)

            x_pred_avg = x_pred/n_rounds
            pred_cur_afterdc_avg = pred_cur_afterdc/n_rounds

            psnr_img = []
            for i in range(np.array(x_pred_avg).shape[0]):
                psnr_perimg = psnr(np.array(abs(im_gnd_show[i]))/np.array(abs(im_gnd_show[i])).max(),np.array(abs(show_img_after_dc[i])) / np.array(abs(show_img_after_dc[i])).max())
                psnr_img.append(psnr_perimg)
            print('PSNR per img', np.mean(psnr_img))
            one_sample.append(x_pred_avg)
            one_sample_afterdc.append(pred_cur_afterdc_avg)
            gt_sample.append(gt_mask)
            und_sample.append(im_und)
          one_sample1 = np.array(one_sample)
          one_sample1 = np.transpose(one_sample1, (1,2,0,3,4))

          und_sample1 = np.array(und_sample)
          und_sample1 = np.transpose(und_sample1, (1,2,0,3,4))
          
          gt_sample1 = np.array(gt_sample)
          gt_sample1 = np.transpose(gt_sample1, (1,2,0,3,4))
            
          one_sample_afterdc1 = np.array(one_sample_afterdc)
          one_sample_afterdc1 = np.transpose(one_sample_afterdc1, (1,2,0,3,4)) 
            
          print(one_sample1.shape, 'one_sample1', gt_sample1.shape, 'gt_sample1')
          one_sample_complex = one_sample1[:,0,...] + 1j*one_sample1[:,1,...]
          gt_sample_complex = gt_sample1[:,0,...] + 1j*gt_sample1[:,1,...]
          und_sample_complex = und_sample1[:,0,...] + 1j*und_sample1[:,1,...]
          one_sample_afterdc_complex = one_sample_afterdc1[:,0,...] + 1j*one_sample_afterdc1[:,1,...]
            
          print(one_sample_complex.shape, 'one_sample1', gt_sample_complex.shape, 'gt_sample1')
          np.savez(npz_save_path+f'{sample_index}_results.npz', gnd=np.array(gt_sample_complex), und=np.array(und_sample_complex), pred = np.array(one_sample_afterdc_complex))
          psnr_ = psnr(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_complex)) / np.array(abs(one_sample_complex)).max())
          print('onepsnr', psnr_,)
          psnr_afterdc = psnr(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_afterdc_complex)) / np.array(abs(one_sample_afterdc_complex)).max())
          print('onepsnr_afterdc', psnr_afterdc)
        
          mse_ = mse(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_complex)) / np.array(abs(one_sample_complex)).max())
          mse_afterdc = mse(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_afterdc_complex)) / np.array(abs(one_sample_afterdc_complex)).max())          
            
          nmse_ = nmse(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_complex)) / np.array(abs(one_sample_complex)).max())
          nmse_afterdc = nmse(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_afterdc_complex)) / np.array(abs(one_sample_afterdc_complex)).max())
            
          ssim_ = ssim(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_complex)) / np.array(abs(one_sample_complex)).max())
          ssim_afterdc = ssim(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_afterdc_complex)) / np.array(abs(one_sample_afterdc_complex)).max())
            
          print('onessim',ssim_,)
          print('onessim after dc',ssim_afterdc)
            
          psnr_list.append(psnr_)
          mse_list.append(mse_)
          nmse_list.append(nmse_)
          ssim_list.append(ssim_)
          psnr_list_adc.append(psnr_afterdc)
          mse_list_adc.append(mse_afterdc)
          nmse_list_adc.append(nmse_afterdc)
          ssim_list_adc.append(ssim_afterdc)
        
        
          logger.info(
              f"{sample_index} n round psnr {psnr_list[-1]}, ssim {ssim_list[-1]}, mse {mse_list[-1]}, nmse {nmse_list[-1]}")
          logger.info(
              f"{sample_index} n round after dc psnr {psnr_list_adc[-1]}, ssim {ssim_list_adc[-1]}, mse {mse_list_adc[-1]}, nmse {nmse_list_adc[-1]}")
        
          print("current avg PSNR:", np.mean(psnr_list), 'current avg after dc PSNR:',np.mean(psnr_list_adc))
          print("current avg SSIM:", np.mean(ssim_list), 'current avg after dc SSIM:',np.mean(ssim_list_adc))
          print("current avg NMSE:", np.mean(nmse_list), 'current avg after dc NMSE:',np.mean(nmse_list_adc))

    my_length = len(psnr_list)
    print('psnr_lenth', my_length)
    logger.info("measure total avg")
    logger.info(f"mean psnr {np.mean(psnr_list)}, std psnr {np.std(psnr_list)}")
    logger.info(f"mean ssim {np.mean(ssim_list)}, std ssim {np.std(ssim_list)}")
    logger.info(f"mean mse {np.mean(mse_list)}, std mse {np.std(mse_list)}")
    logger.info(f"mean nmse {np.mean(nmse_list)}, std nmse {np.std(nmse_list)}")
    logger.info("measure total avg after dc")
    logger.info(f"mean psnr {np.mean(psnr_list_adc)}, std psnr {np.std(psnr_list_adc)}")
    logger.info(f"mean ssim {np.mean(ssim_list_adc)}, std ssim {np.std(ssim_list_adc)}")
    logger.info(f"mean mse {np.mean(mse_list_adc)}, std mse {np.std(mse_list_adc)}")
    logger.info(f"mean nmse {np.mean(nmse_list_adc)}, std nmse {np.std(nmse_list_adc)}")


def CMR_sampling_Multi_onecoil(diffusion_model, ddp_model, output_folder, dataset, logger, clip_denoised, pic_save_path,npz_save_path):
    if not os.path.exists(pic_save_path):
        os.makedirs(pic_save_path)
    ddp_model.eval()
    os.makedirs(npz_save_path, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    n_rounds = 1
    ssim_list, mse_list, nmse_list, psnr_list = [],[],[],[]
    ssim_list_adc, mse_list_adc, nmse_list_adc, psnr_list_adc = [],[],[],[]
    logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  
    filename=output_folder+'/combined result.log',  
    filemode='w'  # 文件模式，'a' 代表追加模式，'w' 代表覆盖模式
)
    logger_com = logging.getLogger(__name__)
    with torch.no_grad():
        data_idx_list = [i for i in range(len(dataset))]

        ssim_coil, mse_coil, nmse_coil, psnr_coil = [],[],[],[]
        ssim_coil_adc, mse_coil_adc, nmse_coil_adc, psnr_coil_adc = [],[],[],[]
        pred_eachcoil = []
        pred_afdc_eachcoil = []
        gnd_eachcoil = []
        und_eachcoil = []
        for data_idx in data_idx_list:

            one_coil = []
            gt_coil = []
            und_coil = []
            one_coil_afterdc = []
            for idx, slices in enumerate(dataset[data_idx]):
                sample_index = dataset.all_patient[data_idx]
                im_und, im_gnd, mask = slices
                print(im_gnd.shape, im_und.shape)
                gt_mask = im_gnd
                condition_on = im_und
                former_frame_for_feature_extraction = condition_on.to(dist_util.dev())
                x_pred = 0
                pred_cur_afterdc = 0
                for round_index in range(n_rounds):
                    print(f"Current Round: {round_index+1} / Total Round: {n_rounds}")
                    model_kwargs = {
                      "conditioned_image": former_frame_for_feature_extraction}
                    x = diffusion_model.ddim_sample_loop(
                      ddp_model,
                      (former_frame_for_feature_extraction.shape[0], gt_mask.shape[1], former_frame_for_feature_extraction.shape[2],
                          former_frame_for_feature_extraction.shape[3]),
                      progress=True,
                      clip_denoised=clip_denoised,
                      model_kwargs=model_kwargs
                  )
                    pred_cur = x.cpu()
              
                    im_gnd_show = im_gnd[:,0,...]+1j*im_gnd[:,1,...]
              # plt.imsave(pic_save_path + f'im_gnd_sample{sample_index}_slice{idx}.png',abs(im_gnd_show[0,...]), cmap='gray')
                    im_und_show = im_und[:,0,...]+1j*im_und[:,1,...]
              # plt.imsave(pic_save_path + f'im_und_sample{sample_index}_slice{idx}.png',abs(im_und_show[0,...]), cmap='gray')

                    show_img_before_dc = pred_cur[:,0,...]+1j*pred_cur[:,1,...]
              # plt.imsave(pic_save_path + f'before_dc_sample{sample_index}_slice{idx}.png',abs(show_img_before_dc[0,...]), cmap='gray')
                    psnr_be = psnr(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_before_dc)) / np.array(abs(show_img_before_dc)).max())
                    ssim_be = ssim(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_before_dc)) / np.array(abs(show_img_before_dc)).max())
              # print(f'slice {idx} before dc', psnr_be, 'psnr', ssim_be, 'ssim')
#               print(pred_cur.shape, im_und.shape, mask.shape)
              # print('pred scale',abs(pred_cur).max(), abs(pred_cur).min(), abs(pred_cur).mean(),'normed',abs(pred_cur*gnd_norm_para).max())
              # print('und scale',abs(im_und).max(), abs(im_und).min(), abs(im_und).mean(),'normed',abs(im_und*und_norm_para).max())
              # for factor in np.arange(0.60, 0.65, 0.002):
                
                    pred_cur_afterdc = data_consistency(pred_cur, im_und, mask)
                #  print('gnd norm',gnd_norm_para, 'und norm',und_norm_para)
                    show_img_after_dc = pred_cur_afterdc[:,0,...]+1j*pred_cur_afterdc[:,1,...]
                    plt.imsave(pic_save_path + f'after_dc_sample{sample_index}_slice{idx}.png',abs(show_img_after_dc[0,...]), cmap='gray')
              
                    psnr_af = psnr(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_after_dc)) / np.array(abs(show_img_after_dc)).max())
                    ssim_af = ssim(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_after_dc)) / np.array(abs(show_img_after_dc)).max())
                    print(f' slice {idx} after dc',psnr_af, 'psnr', ssim_af, 'ssim')
               
                    error = abs(im_gnd_show[0,...])-abs(show_img_before_dc[0,...])
                    plt.imsave(pic_save_path + f'error_sample{sample_index}_slice{idx}.png',error, cmap='gray')

                    x_pred = x_pred + np.array(pred_cur)
                    pred_cur_afterdc = pred_cur_afterdc + np.array(pred_cur_afterdc)

                x_pred_avg = x_pred/n_rounds
                pred_cur_afterdc_avg = pred_cur_afterdc/n_rounds

            # psnr_img = []
            # for i in range(np.array(x_pred_avg).shape[0]):
            #     psnr_perimg = psnr(np.array(abs(im_gnd_show[i]))/np.array(abs(im_gnd_show[i])).max(),np.array(abs(show_img_after_dc[i])) / np.array(abs(show_img_after_dc[i])).max())
            #     psnr_img.append(psnr_perimg)
            # print('PSNR per img', np.mean(psnr_img))
                one_coil.append(x_pred_avg)
                one_coil_afterdc.append(pred_cur_afterdc_avg)
                gt_coil.append(gt_mask)
                und_coil.append(im_und)
            one_coil1 = np.array(one_coil)
            one_coil1 = np.transpose(one_coil1, (1,2,0,3,4))
            gt_coil1 = np.array(gt_coil)
            gt_coil1 = np.transpose(gt_coil1, (1,2,0,3,4))
            und_coil1 = np.array(und_coil)
            und_coil1 = np.transpose(und_coil1, (1,2,0,3,4))            
            one_coil_afterdc1 = np.array(one_coil_afterdc)
            one_coil_afterdc1 = np.transpose(one_coil_afterdc1, (1,2,0,3,4)) 
            
            one_sample_complex = one_coil1[:,0,...] + 1j*one_coil1[:,1,...]
            gt_sample_complex = gt_coil1[:,0,...] + 1j*gt_coil1[:,1,...]
            und_sample_complex = und_coil1[:,0,...] + 1j*und_coil1[:,1,...]
            one_sample_afterdc_complex = one_coil_afterdc1[:,0,...] + 1j*one_coil_afterdc1[:,1,...]
            
            pred_eachcoil.append(one_sample_complex)
            pred_afdc_eachcoil.append(one_sample_afterdc_complex)
            gnd_eachcoil.append(gt_sample_complex)
            und_eachcoil.append(und_sample_complex)

            print(one_sample_complex.shape, 'one_sample1', gt_sample_complex.shape, 'gt_sample1')
            psnr_ = psnr(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_complex)) / np.array(abs(one_sample_complex)).max())
            psnr_coil.append(psnr_)
            print('onecoil', psnr_,)
            psnr_afterdc = psnr(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_afterdc_complex)) / np.array(abs(one_sample_afterdc_complex)).max())
            psnr_coil_adc.append(psnr_afterdc)
            print('onecoilpsnr_afterdc', psnr_afterdc)
        
            mse_ = mse(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_complex)) / np.array(abs(one_sample_complex)).max())
            mse_afterdc = mse(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_afterdc_complex)) / np.array(abs(one_sample_afterdc_complex)).max())          
            mse_coil.append(mse_)
            mse_coil_adc.append(mse_afterdc)
            
            nmse_ = nmse(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_complex)) / np.array(abs(one_sample_complex)).max())
            nmse_afterdc = nmse(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_afterdc_complex)) / np.array(abs(one_sample_afterdc_complex)).max())
            nmse_coil.append(nmse_)
            nmse_coil_adc.append(nmse_afterdc)
           
            ssim_ = ssim(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_complex)) / np.array(abs(one_sample_complex)).max())
            ssim_afterdc = ssim(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_afterdc_complex)) / np.array(abs(one_sample_afterdc_complex)).max())
            ssim_coil.append(ssim_)
            ssim_coil_adc.append(ssim_afterdc)
          
            logger.info(
              f"{sample_index} coil{(data_idx+1)%10} psnr {psnr_coil[-1]}, ssim {ssim_coil[-1]}, mse {mse_coil[-1]}, nmse {nmse_coil[-1]}")
            logger.info(
              f"{sample_index} coil{(data_idx+1)%10} after dc psnr {psnr_coil_adc[-1]}, ssim {ssim_coil_adc[-1]}, mse {mse_coil_adc[-1]}, nmse {nmse_coil_adc[-1]}")
            print(len(pred_eachcoil))
            if len(pred_eachcoil) == 10:
                combined_pred = coil_combination_rss(pred_eachcoil)
                combined_gnd = coil_combination_rss(gnd_eachcoil)
                combined_pred_adc = coil_combination_rss(pred_afdc_eachcoil)
                combined_und = coil_combination_rss(und_eachcoil)
                np.savez(npz_save_path+f'{sample_index}_combined_results.npz', gnd=np.array(combined_gnd), und=np.array(combined_und), pred = np.array(combined_pred_adc))

                for i in range(combined_pred_adc.shape[1]):
                    plt.imsave(pic_save_path + f'after_dc_combined_sample{sample_index}_slice{i}.png',abs(combined_pred_adc[0,i,...]), cmap='gray')
                    plt.imsave(pic_save_path + f'und_combined_sample{sample_index}_slice{i}.png',abs(combined_und[0,i,...]), cmap='gray')
                    plt.imsave(pic_save_path + f'gnd_combined_sample{sample_index}_slice{i}.png',abs(combined_gnd[0,i,...]), cmap='gray')
                
                
                psnr_value, ssim_value, nmse_value, mse_value = cal_four_metrics(combined_gnd, combined_pred)
                psnr_valuedc, ssim_valuedc, nmse_valuedc, mse_valuedc = cal_four_metrics(combined_gnd, combined_pred_adc)

                logger.info(
                 f"{sample_index} combined psnr {psnr_value}, ssim {ssim_value}, nmse {nmse_value}, mse {mse_value}")
                logger.info(
                 f"{sample_index} combined after dc psnr {psnr_valuedc}, ssim {ssim_valuedc}, nmse {nmse_valuedc}, mse {mse_valuedc}")
                logger.info('==============================================NEXT SAMPLE===============================================================')
                psnr_list.append(psnr_value)
                mse_list.append(mse_value)
                nmse_list.append(nmse_value)
                ssim_list.append(ssim_value)
                psnr_list_adc.append(psnr_valuedc)
                mse_list_adc.append(mse_valuedc)
                nmse_list_adc.append(nmse_valuedc)
                ssim_list_adc.append(ssim_valuedc)
                logger_com.info(f'PSNR list {psnr_list_adc}')
                logger_com.info(f'MSE list {mse_list_adc}')
                logger_com.info(f'SSIM list {ssim_list_adc}')
                logger_com.info(f'NMSE list {nmse_list_adc}')
                pred_eachcoil = []
                pred_afdc_eachcoil = []
                gnd_eachcoil = []
                und_eachcoil = []

                print("current avg PSNR:", np.mean(psnr_list), 'current avg after dc PSNR:',np.mean(psnr_list_adc))
                print("current avg SSIM:", np.mean(ssim_list), 'current avg after dc SSIM:',np.mean(ssim_list_adc))
                print("current avg NMSE:", np.mean(nmse_list), 'current avg after dc NMSE:',np.mean(nmse_list_adc))

    my_length = len(psnr_list)
    print('psnr_lenth', my_length)
    logger.info("=====================measure total avg MULTI================================")
    logger.info(f"mean psnr {np.mean(psnr_list)}, std psnr {np.std(psnr_list)}")
    logger.info(f"mean ssim {np.mean(ssim_list)}, std ssim {np.std(ssim_list)}")
    logger.info(f"mean mse {np.mean(mse_list)}, std mse {np.std(mse_list)}")
    logger.info(f"mean nmse {np.mean(nmse_list)}, std nmse {np.std(nmse_list)}")
    logger.info("========================measure total avg after dc MULTI==============================")
    logger.info(f"mean psnr {np.mean(psnr_list_adc)}, std psnr {np.std(psnr_list_adc)}")
    logger.info(f"mean ssim {np.mean(ssim_list_adc)}, std ssim {np.std(ssim_list_adc)}")
    logger.info(f"mean mse {np.mean(mse_list_adc)}, std mse {np.std(mse_list_adc)}")
    logger.info(f"mean nmse {np.mean(nmse_list_adc)}, std nmse {np.std(nmse_list_adc)}")


def CMR_sampling_Multi_with_Sensdc(diffusion_model, ddp_model, output_folder, dataset, logger, clip_denoised, pic_save_path):
    if not os.path.exists(pic_save_path):
        os.makedirs(pic_save_path)
    ddp_model.eval()
    os.makedirs(output_folder, exist_ok=True)
    n_rounds = 1
    print(n_rounds,'nnnn')
    ssim_list, mse_list, nmse_list, psnr_list = [],[],[],[]
    ssim_list_adc, mse_list_adc, nmse_list_adc, psnr_list_adc = [],[],[],[]
    with torch.no_grad():
      with open('/home/zhenlin/norm_test.log', 'r') as file:
          file_content = file.readlines()
      data_idx_list = [i for i in range(len(dataset))]
      # data_idx_list = [0]
      for data_idx in data_idx_list:
          gnd_norm_para = torch.tensor(float(file_content[data_idx].split()[5]))
          und_norm_para = torch.tensor(float(file_content[data_idx].split()[11]))
          sample_index = file_content[data_idx].split()[1]
          one_sample = []
          gt_sample = []
          sens_sample = []
          mask_sample = []
          und_sample = []
          for idx, slices in enumerate(dataset[data_idx]):
            im_und, im_gnd, mask, sens = slices
            print(im_gnd.shape, im_und.shape)
            gt_mask = im_gnd
            condition_on = im_und
            former_frame_for_feature_extraction = condition_on.to(dist_util.dev())
            x_pred = 0
            for round_index in range(n_rounds):
                print(f"Current Round: {round_index+1} / Total Round: {n_rounds}")
                model_kwargs = {
                  "conditioned_image": former_frame_for_feature_extraction}
                x = diffusion_model.ddim_sample_loop(
                  ddp_model,
                  (former_frame_for_feature_extraction.shape[0], gt_mask.shape[1], former_frame_for_feature_extraction.shape[2],
                      former_frame_for_feature_extraction.shape[3]),
                  progress=True,
                  clip_denoised=clip_denoised,
                  model_kwargs=model_kwargs
              )
                pred_cur = x.cpu()
              
                im_gnd_show = im_gnd[:,0,...]+1j*im_gnd[:,1,...]
                plt.imsave(pic_save_path + f'im_gnd_sample{sample_index}_slice{idx}.png',abs(im_gnd_show[0,...]), cmap='gray')
                im_und_show = im_und[:,0,...]+1j*im_und[:,1,...]
                plt.imsave(pic_save_path + f'im_und_sample{sample_index}_slice{idx}.png',abs(im_und_show[0,...]), cmap='gray')

                show_img_before_dc = pred_cur[:,0,...]+1j*pred_cur[:,1,...]
                plt.imsave(pic_save_path + f'before_dc_sample{sample_index}_slice{idx}.png',abs(show_img_before_dc[0,...]), cmap='gray')
                psnr_be = psnr(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_before_dc)) / np.array(abs(show_img_before_dc)).max())
                ssim_be = ssim(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_before_dc)) / np.array(abs(show_img_before_dc)).max())
                print(f'slice {idx} before dc', psnr_be, 'psnr', ssim_be, 'ssim')
                print('pred scale',abs(pred_cur).max(), abs(pred_cur).min(), abs(pred_cur).mean(),'normed',abs(pred_cur*gnd_norm_para).max())
                print('und scale',abs(im_und).max(), abs(im_und).min(), abs(im_und).mean(),'normed',abs(im_und*und_norm_para).max())
                
              # pred_cur_afterdc = data_consistency_Sens(pred_cur, im_und, mask)
              # show_img_after_dc = pred_cur_afterdc[:,0,...]+1j*pred_cur_afterdc[:,1,...]
              # plt.imsave(pic_save_path + f'after_dc_sample{sample_index}_slice{idx}.png',abs(show_img_after_dc[0,...]), cmap='gray')
              
              # psnr_af = psnr(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_after_dc)) / np.array(abs(show_img_after_dc)).max())
              # ssim_af = ssim(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_after_dc)) / np.array(abs(show_img_after_dc)).max())
              # print(f' slice {idx} after dc',psnr_af, 'psnr', ssim_af, 'ssim')
              
              # error = abs(im_gnd_show[0,...])-abs(show_img_before_dc[0,...])
              # plt.imsave(pic_save_path + f'error_sample{sample_index}_slice{idx}.png',error, cmap='gray')

                x_pred = x_pred + np.array(pred_cur)
                # pred_cur_afterdc = pred_cur_afterdc + np.array(pred_cur_afterdc)

            x_pred_avg = x_pred/n_rounds
            # pred_cur_afterdc_avg = pred_cur_afterdc/n_rounds

            one_sample.append(x_pred_avg)
            sens_sample.append(sens)
            mask_sample.append(mask)
            # one_sample_afterdc.append(pred_cur_afterdc_avg)
            gt_sample.append(gt_mask)
            und_sample.append(im_und)
          
          #from [s, t, c, h, w] to [t,c,s,h,w] then [t,s,h,w]
          one_sample1 = np.array(one_sample)
          one_sample1 = np.transpose(one_sample1, (1,2,0,3,4))
          one_sample = one_sample1[:,0,...] + 1j*one_sample1[:,1,...]
            
          gt_sample1 = np.array(gt_sample)
          gt_sample1 = np.transpose(gt_sample1, (1,2,0,3,4))
          gt_sample = gt_sample1[:,0,...] + 1j*gt_sample1[:,1,...]

          und_sample1 = np.array(und_sample)
          und_sample1 = np.transpose(und_sample1, (1,2,0,3,4))   
          und_sample = und_sample1[:,0,...] + 1j*und_sample1[:,1,...]

          # for sens and mask
          sens_sample1 = np.array(sens_sample)  # to [s,coil,h,w]
          # sens_sample1 = np.transpose(sens_sample1, (1,2,0,3,4))
          # one_sample_complex = one_sample1[:,0,...] + 1j*one_sample1[:,1,...]

          mask_sample1 = np.array(mask_sample)
          mask_sample = np.transpose(mask_sample1, (1,2,0,3,4))[:,0,...] #to [t,s,h,w]
                   
          print(one_sample1.shape, 'one_sample1', gt_sample1.shape, 'gt_sample1')
          # one_sample_complex = one_sample1[:,0,...] + 1j*one_sample1[:,1,...]
          # gt_sample_complex = gt_sample1[:,0,...] + 1j*gt_sample1[:,1,...]
            
          pred_cur_afterdc = data_consistency_Sens(one_sample, und_sample, mask_sample, sens_sample1)
          # pred_cur_afterdc_norm = data_consistency_Sens_slicenorm(pred_cur_dc, im_und_dc, mask, sens_sample1)
          print(one_sample.shape, 'one_sample1', gt_sample.shape, 'gt_sample1')
          psnr_ = psnr(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(one_sample)) / np.array(abs(one_sample)).max())
          print('onepsnr', psnr_,)
          psnr_afterdc = psnr(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(pred_cur_afterdc)) / np.array(abs(pred_cur_afterdc)).max())
          print('onepsnr_afterdc', psnr_afterdc)
        
          mse_ = mse(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(one_sample)) / np.array(abs(one_sample)).max())
          mse_afterdc = mse(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(pred_cur_afterdc)) / np.array(abs(pred_cur_afterdc)).max())          
            
          nmse_ = nmse(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(one_sample)) / np.array(abs(one_sample)).max())
          nmse_afterdc = nmse(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(pred_cur_afterdc)) / np.array(abs(pred_cur_afterdc)).max())
            
          ssim_ = ssim(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(one_sample)) / np.array(abs(one_sample)).max())
          ssim_afterdc = ssim(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(pred_cur_afterdc)) / np.array(abs(pred_cur_afterdc)).max())
          
            
          print('onessim',ssim_,)
          print('onessim after dc',ssim_afterdc)
            
          psnr_list.append(psnr_)
          mse_list.append(mse_)
          nmse_list.append(nmse_)
          ssim_list.append(ssim_)
          psnr_list_adc.append(psnr_afterdc)
          mse_list_adc.append(mse_afterdc)
          nmse_list_adc.append(nmse_afterdc)
          ssim_list_adc.append(ssim_afterdc)
        
          logger.info(
              f"{sample_index} n round psnr {psnr_list[-1]}, ssim {ssim_list[-1]}, mse {mse_list[-1]}, nmse {nmse_list[-1]}")
          logger.info(
              f"{sample_index} n round after dc psnr {psnr_list_adc[-1]}, ssim {ssim_list_adc[-1]}, mse {mse_list_adc[-1]}, nmse {nmse_list_adc[-1]}")
        
          print("current avg PSNR:", np.mean(psnr_list), 'current avg after dc PSNR:',np.mean(psnr_list_adc))
          print("current avg SSIM:", np.mean(ssim_list), 'current avg after dc SSIM:',np.mean(ssim_list_adc))
          print("current avg NMSE:", np.mean(nmse_list), 'current avg after dc NMSE:',np.mean(nmse_list_adc))

    my_length = len(psnr_list)
    print('psnr_lenth', my_length)
    logger.info("measure total avg")
    logger.info(f"mean psnr {np.mean(psnr_list)}, std psnr {np.std(psnr_list)}")
    logger.info(f"mean ssim {np.mean(ssim_list)}, std ssim {np.std(ssim_list)}")
    logger.info(f"mean mse {np.mean(mse_list)}, std mse {np.std(mse_list)}")
    logger.info(f"mean nmse {np.mean(nmse_list)}, std nmse {np.std(nmse_list)}")
    logger.info("measure total avg after dc")
    logger.info(f"mean psnr {np.mean(psnr_list_adc)}, std psnr {np.std(psnr_list_adc)}")
    logger.info(f"mean ssim {np.mean(ssim_list_adc)}, std ssim {np.std(ssim_list_adc)}")
    logger.info(f"mean mse {np.mean(mse_list_adc)}, std mse {np.std(mse_list_adc)}")
    logger.info(f"mean nmse {np.mean(nmse_list_adc)}, std nmse {np.std(nmse_list_adc)}")



def CMR_sampling_Multi_with_SensSlicedc(diffusion_model, ddp_model, output_folder, dataset, logger, clip_denoised, pic_save_path, npz_save_path):
    if not os.path.exists(pic_save_path):
        os.makedirs(pic_save_path)
    ddp_model.eval()
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(npz_save_path, exist_ok=True)
    n_rounds = 1
    print(n_rounds,'nnnn')
    ssim_list, mse_list, nmse_list, psnr_list = [],[],[],[]
    ssim_list_adc, mse_list_adc, nmse_list_adc, psnr_list_adc = [],[],[],[]
    with torch.no_grad():
      with open('/home/zhenlin/norm_test.log', 'r') as file:
          file_content = file.readlines()
      data_idx_list = [i for i in range(len(dataset))]
      # data_idx_list = [0]
      for data_idx in data_idx_list:
          gnd_norm_para = torch.tensor(float(file_content[data_idx].split()[5]))
          und_norm_para = torch.tensor(float(file_content[data_idx].split()[11]))
          sample_index = file_content[data_idx].split()[1]
          one_sample = []
          gt_sample = []
          sens_sample = []
          mask_sample = []
          und_sample = []
          one_sample_afterdc = []
          for idx, slices in enumerate(dataset[data_idx]):
            im_und, im_gnd, mask, sens, im_und2 = slices
            print(im_gnd.shape, im_und.shape)
            gt_mask = im_gnd
            condition_on = im_und
            former_frame_for_feature_extraction = condition_on.to(dist_util.dev())
            x_pred = 0
            pred_cur_afterdc1 = 0
            for round_index in range(n_rounds):
                print(f"Current Round: {round_index+1} / Total Round: {n_rounds}")
                model_kwargs = {
                  "conditioned_image": former_frame_for_feature_extraction}
                x = diffusion_model.ddim_sample_loop(
                  ddp_model,
                  (former_frame_for_feature_extraction.shape[0], gt_mask.shape[1], former_frame_for_feature_extraction.shape[2],
                      former_frame_for_feature_extraction.shape[3]),
                  progress=True,
                  clip_denoised=clip_denoised,
                  model_kwargs=model_kwargs
              )
                pred_cur = x.cpu()
              
                im_gnd_show = im_gnd[:,0,...]+1j*im_gnd[:,1,...]
                plt.imsave(pic_save_path + f'im_gnd_sample{sample_index}_slice{idx}.png',abs(im_gnd_show[0,...]), cmap='gray')
                im_und_show = im_und[:,0,...]+1j*im_und[:,1,...]
                plt.imsave(pic_save_path + f'im_und_sample{sample_index}_slice{idx}.png',abs(im_und_show[0,...]), cmap='gray')

                show_img_before_dc = pred_cur[:,0,...]+1j*pred_cur[:,1,...]
                plt.imsave(pic_save_path + f'before_dc_sample{sample_index}_slice{idx}.png',abs(show_img_before_dc[0,...]), cmap='gray')
                psnr_be = psnr(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_before_dc)) / np.array(abs(show_img_before_dc)).max())
                ssim_be = ssim(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_before_dc)) / np.array(abs(show_img_before_dc)).max())
                print(f'slice {idx} before dc', psnr_be, 'psnr', ssim_be, 'ssim')
                print('pred scale',abs(pred_cur).max(), abs(pred_cur).min(), abs(pred_cur).mean(),'normed',abs(pred_cur*gnd_norm_para).max())
                print('und scale',abs(im_und).max(), abs(im_und).min(), abs(im_und).mean(),'normed',abs(im_und*und_norm_para).max())
                
                pred_cur_dc = pred_cur[:,0:1,...]+1j*pred_cur[:,1:2,...]
                im_und_dc = im_und[:,0:1,...]+1j*im_und[:,1:2,...]
                pred_cur_afterdc = data_consistency_Sens_slice(pred_cur_dc, im_und2, mask, np.array(sens))
                # pred_cur_afterdc_norm = data_consistency_Sens_slicenorm(pred_cur_dc, im_und_dc, mask, np.array(sens))
                print(pred_cur_afterdc.shape,'afdc shape')
                show_img_after_dc = pred_cur_afterdc[:,0,...]
                # show_img_after_dc_norm = pred_cur_afterdc_norm[:,0,...]
                plt.imsave(pic_save_path + f'after_dc_sample{sample_index}_slice{idx}.png',abs(show_img_after_dc[0,...]), cmap='gray')
              
                psnr_af = psnr(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_after_dc)) / np.array(abs(show_img_after_dc)).max())
                ssim_af = ssim(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_after_dc)) / np.array(abs(show_img_after_dc)).max())

                # psnr_afn = psnr(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_after_dc_norm)) / np.array(abs(show_img_after_dc_norm)).max())
                # ssim_afn = ssim(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_after_dc_norm)) / np.array(abs(show_img_after_dc_norm)).max())
                
                
                print(f' slice {idx} after dc',psnr_af, 'psnr', ssim_af, 'ssim')
                # print(f' slice {idx} after dc norm',psnr_afn, 'psnr', ssim_afn, 'ssim')
              
                error = abs(im_gnd_show[0,...])-abs(show_img_before_dc[0,...])
                plt.imsave(pic_save_path + f'error_sample{sample_index}_slice{idx}.png',error, cmap='gray')

                x_pred = x_pred + np.array(pred_cur)
                pred_cur_afterdc1 = pred_cur_afterdc1 + np.array(pred_cur_afterdc)

            x_pred_avg = x_pred/n_rounds
            pred_cur_afterdc_avg = pred_cur_afterdc1/n_rounds

            one_sample.append(x_pred_avg)
            one_sample_afterdc.append(pred_cur_afterdc_avg[:,0,...])
            gt_sample.append(gt_mask)
            und_sample.append(im_und)
          
          #from [s, t, c, h, w] to [t,c,s,h,w] then [t,s,h,w]
          one_sample1 = np.array(one_sample)
          one_sample1 = np.transpose(one_sample1, (1,2,0,3,4))
          one_sample = one_sample1[:,0,...] + 1j*one_sample1[:,1,...]
            
          one_sample_afterdc = np.array(one_sample_afterdc)
          pred_cur_afterdc = np.transpose(one_sample_afterdc, (1,0,2,3))         
          
            
          gt_sample1 = np.array(gt_sample)
          gt_sample1 = np.transpose(gt_sample1, (1,2,0,3,4))
          gt_sample = gt_sample1[:,0,...] + 1j*gt_sample1[:,1,...]

          und_sample1 = np.array(und_sample)
          und_sample1 = np.transpose(und_sample1, (1,2,0,3,4))   
          und_sample = und_sample1[:,0,...] + 1j*und_sample1[:,1,...]

          np.savez(npz_save_path + f'{sample_index}_result.npz', pred=pred_cur_afterdc, und=und_sample, gnd = gt_sample)

                   
          # print(one_sample1.shape, 'one_sample1', gt_sample1.shape, 'gt_sample1')
          # one_sample_complex = one_sample1[:,0,...] + 1j*one_sample1[:,1,...]
          # gt_sample_complex = gt_sample1[:,0,...] + 1j*gt_sample1[:,1,...]
            
          # pred_cur_afterdc = data_consistency_Sens(one_sample, und_sample, mask_sample, sens_sample1)
          print(one_sample.shape, 'one_sample1', gt_sample.shape, 'gt_sample1', pred_cur_afterdc.shape)
          psnr_ = psnr(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(one_sample)) / np.array(abs(one_sample)).max())
          print('onepsnr', psnr_,)
          psnr_afterdc = psnr(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(pred_cur_afterdc)) / np.array(abs(pred_cur_afterdc)).max())
          # print('onepsnr_afterdc', psnr_afterdc)
        
          mse_ = mse(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(one_sample)) / np.array(abs(one_sample)).max())
          mse_afterdc = mse(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(pred_cur_afterdc)) / np.array(abs(pred_cur_afterdc)).max())          
            
          nmse_ = nmse(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(one_sample)) / np.array(abs(one_sample)).max())
          nmse_afterdc = nmse(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(pred_cur_afterdc)) / np.array(abs(pred_cur_afterdc)).max())
            
          ssim_ = ssim(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(one_sample)) / np.array(abs(one_sample)).max())
          ssim_afterdc = ssim(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(pred_cur_afterdc)) / np.array(abs(pred_cur_afterdc)).max())
          
            
          print('onessim',ssim_,)
          print('onessim after dc',ssim_afterdc)
            
          psnr_list.append(psnr_)
          mse_list.append(mse_)
          nmse_list.append(nmse_)
          ssim_list.append(ssim_)
          psnr_list_adc.append(psnr_afterdc)
          mse_list_adc.append(mse_afterdc)
          nmse_list_adc.append(nmse_afterdc)
          ssim_list_adc.append(ssim_afterdc)
        
          logger.info(
              f"{sample_index} n round psnr {psnr_list[-1]}, ssim {ssim_list[-1]}, mse {mse_list[-1]}, nmse {nmse_list[-1]}")
          logger.info(
              f"{sample_index} n round after dc psnr {psnr_list_adc[-1]}, ssim {ssim_list_adc[-1]}, mse {mse_list_adc[-1]}, nmse {nmse_list_adc[-1]}")
        
          print("current avg PSNR:", np.mean(psnr_list), 'current avg after dc PSNR:',np.mean(psnr_list_adc))
          print("current avg SSIM:", np.mean(ssim_list), 'current avg after dc SSIM:',np.mean(ssim_list_adc))
          print("current avg NMSE:", np.mean(nmse_list), 'current avg after dc NMSE:',np.mean(nmse_list_adc))

    my_length = len(psnr_list)
    print('psnr_lenth', my_length)
    logger.info("measure total avg")
    logger.info(f"mean psnr {np.mean(psnr_list)}, std psnr {np.std(psnr_list)}")
    logger.info(f"mean ssim {np.mean(ssim_list)}, std ssim {np.std(ssim_list)}")
    logger.info(f"mean mse {np.mean(mse_list)}, std mse {np.std(mse_list)}")
    logger.info(f"mean nmse {np.mean(nmse_list)}, std nmse {np.std(nmse_list)}")
    logger.info("measure total avg after dc")
    logger.info(f"mean psnr {np.mean(psnr_list_adc)}, std psnr {np.std(psnr_list_adc)}")
    logger.info(f"mean ssim {np.mean(ssim_list_adc)}, std ssim {np.std(ssim_list_adc)}")
    logger.info(f"mean mse {np.mean(mse_list_adc)}, std mse {np.std(mse_list_adc)}")
    logger.info(f"mean nmse {np.mean(nmse_list_adc)}, std nmse {np.std(nmse_list_adc)}")




def CMR_sampling_func_with_dc_intraining(diffusion_model, ddp_model, output_folder, dataset, step, clip_denoised):
    logging.basicConfig(filename=f'{output_folder}/train_result.log', level=logging.INFO)
    pic_save_path = os.path.join(output_folder,f'Pic_result_{step}/')
    if not os.path.exists(pic_save_path):
        os.makedirs(pic_save_path)
    ddp_model.eval()
    os.makedirs(output_folder, exist_ok=True)
    n_rounds = 1
    print(n_rounds,'nnnn')
    ssim_list, mse_list, nmse_list, psnr_list = [],[],[],[]
    ssim_list_adc, mse_list_adc, nmse_list_adc, psnr_list_adc = [],[],[],[]
    with torch.no_grad():
      with open('/home/zhenlin/norm_test.log', 'r') as file:
          file_content = file.readlines()
      data_idx_list = [i for i in range(len(dataset))]
      data_idx_list = data_idx_list[::10]
      for data_idx in data_idx_list:
          gnd_norm_para = torch.tensor(float(file_content[data_idx].split()[5]))
          und_norm_para = torch.tensor(float(file_content[data_idx].split()[11]))
          sample_index = file_content[data_idx].split()[1]
          one_sample = []
          gt_sample = []
          one_sample_afterdc = []
          for idx, slices in enumerate(dataset[data_idx]):
              
            im_und, im_gnd, mask = slices
            print(im_gnd.shape, im_und.shape)
            gt_mask = im_gnd
            condition_on = im_und
            former_frame_for_feature_extraction = condition_on.to(dist_util.dev())
            x_pred = 0
            pred_cur_afterdc = 0
            for round_index in range(n_rounds):
              print(f"Current Round: {round_index+1} / Total Round: {n_rounds}")
              model_kwargs = {
                "conditioned_image": former_frame_for_feature_extraction}
              x = diffusion_model.ddim_sample_loop(
                ddp_model,
                (former_frame_for_feature_extraction.shape[0], gt_mask.shape[1], former_frame_for_feature_extraction.shape[2],
                    former_frame_for_feature_extraction.shape[3]),
                progress=True,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs
            )
              pred_cur = x.cpu()
              
              im_gnd_show = im_gnd[:,0,...]+1j*im_gnd[:,1,...]
              plt.imsave(pic_save_path + f'im_gnd_sample{sample_index}_slice{idx}.png',abs(im_gnd_show[0,...]), cmap='gray')
              im_und_show = im_und[:,0,...]+1j*im_und[:,1,...]
              plt.imsave(pic_save_path + f'im_und_sample{sample_index}_slice{idx}.png',abs(im_und_show[0,...]), cmap='gray')

              
              show_img_before_dc = pred_cur[:,0,...]+1j*pred_cur[:,1,...]
              plt.imsave(pic_save_path + f'before_dc_sample{sample_index}_slice{idx}.png',abs(show_img_before_dc[0,...]), cmap='gray')
              psnr_be = psnr(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_before_dc)) / np.array(abs(show_img_before_dc)).max())
              ssim_be = ssim(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_before_dc)) / np.array(abs(show_img_before_dc)).max())
              print(f'slice {idx} before dc', psnr_be, 'psnr', ssim_be, 'ssim')

              pred_cur_afterdc = data_consistency(pred_cur, im_und, mask)

              show_img_after_dc = pred_cur_afterdc[:,0,...]+1j*pred_cur_afterdc[:,1,...]
              plt.imsave(pic_save_path + f'after_dc_sample{sample_index}_slice{idx}.png',abs(show_img_after_dc[0,...]), cmap='gray')
              
              psnr_af = psnr(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_after_dc)) / np.array(abs(show_img_after_dc)).max())
              ssim_af = ssim(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_after_dc)) / np.array(abs(show_img_after_dc)).max())
              print(f'slice {idx} after dc',psnr_af, 'psnr', ssim_af, 'ssim')
              
              error = abs(im_gnd_show[0,...])-abs(show_img_before_dc[0,...])
              plt.imsave(pic_save_path + f'error_sample{sample_index}_slice{idx}.png',error, cmap='gray')

              x_pred = x_pred + np.array(pred_cur)
              pred_cur_afterdc = pred_cur_afterdc + np.array(pred_cur_afterdc)

            x_pred_avg = x_pred/n_rounds
            pred_cur_afterdc_avg = pred_cur_afterdc/n_rounds
            psnr_img = []
            for i in range(np.array(x_pred_avg).shape[0]):
                psnr_perimg = psnr(np.array(abs(im_gnd_show[i]))/np.array(abs(im_gnd_show[i])).max(),np.array(abs(show_img_after_dc[i])) / np.array(abs(show_img_after_dc[i])).max())
                psnr_img.append(psnr_perimg)
            print('PSNR per img', np.mean(psnr_img))
            one_sample.append(x_pred_avg)
            one_sample_afterdc.append(pred_cur_afterdc_avg)
            gt_sample.append(gt_mask)
          one_sample1 = np.array(one_sample)
          one_sample1 = np.transpose(one_sample1, (1,2,0,3,4))
            
          gt_sample1 = np.array(gt_sample)
          gt_sample1 = np.transpose(gt_sample1, (1,2,0,3,4))
            
          one_sample_afterdc1 = np.array(one_sample_afterdc)
          one_sample_afterdc1 = np.transpose(one_sample_afterdc1, (1,2,0,3,4)) 
            
          print(one_sample1.shape, 'one_sample1', gt_sample1.shape, 'gt_sample1')
          one_sample_complex = one_sample1[:,0,...] + 1j*one_sample1[:,1,...]
          gt_sample_complex = gt_sample1[:,0,...] + 1j*gt_sample1[:,1,...]
          one_sample_afterdc_complex = one_sample_afterdc1[:,0,...] + 1j*one_sample_afterdc1[:,1,...]
            
          print(one_sample_complex.shape, 'one_sample1', gt_sample_complex.shape, 'gt_sample1')
          psnr_ = psnr(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_complex)) / np.array(abs(one_sample_complex)).max())
          print('onepsnr', psnr_,)
          psnr_afterdc = psnr(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_afterdc_complex)) / np.array(abs(one_sample_afterdc_complex)).max())
          print('onepsnr_afterdc', psnr_afterdc)
        
          mse_ = mse(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_complex)) / np.array(abs(one_sample_complex)).max())
          mse_afterdc = mse(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_afterdc_complex)) / np.array(abs(one_sample_afterdc_complex)).max())          
            
          nmse_ = nmse(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_complex)) / np.array(abs(one_sample_complex)).max())
          nmse_afterdc = nmse(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_afterdc_complex)) / np.array(abs(one_sample_afterdc_complex)).max())
            
          ssim_ = ssim(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_complex)) / np.array(abs(one_sample_complex)).max())
          ssim_afterdc = ssim(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_afterdc_complex)) / np.array(abs(one_sample_afterdc_complex)).max())
            
          print('onessim',ssim_,)
          print('onessim after dc',ssim_afterdc)
            
          psnr_list.append(psnr_)
          mse_list.append(mse_)
          nmse_list.append(nmse_)
          ssim_list.append(ssim_)
          psnr_list_adc.append(psnr_afterdc)
          mse_list_adc.append(mse_afterdc)
          nmse_list_adc.append(nmse_afterdc)
          ssim_list_adc.append(ssim_afterdc)
        
          
          logging.info(
              f"{sample_index} n round psnr {psnr_list[-1]}, ssim {ssim_list[-1]}, mse {mse_list[-1]}, nmse {nmse_list[-1]}")
          logging.info(
              f"{sample_index} n round after dc psnr {psnr_list_adc[-1]}, ssim {ssim_list_adc[-1]}, mse {mse_list_adc[-1]}, nmse {nmse_list_adc[-1]}")
        
          print("test avg PSNR:", np.mean(psnr_list), 'test avg after dc PSNR:',np.mean(psnr_list_adc))
          print("test avg SSIM:", np.mean(ssim_list), 'test avg after dc SSIM:',np.mean(ssim_list_adc))
          print("test avg NMSE:", np.mean(nmse_list), 'test avg after dc NMSE:',np.mean(nmse_list_adc))

    my_length = len(psnr_list)
    print('psnr_lenth', my_length)
    logging.info(f"step {step}")
    logging.info(f"{step} measure total avg")
    logging.info(f"mean psnr {np.mean(psnr_list)}, std psnr {np.std(psnr_list)}")
    logging.info(f"mean ssim {np.mean(ssim_list)}, std ssim {np.std(ssim_list)}")
    logging.info(f"mean mse {np.mean(mse_list)}, std mse {np.std(mse_list)}")
    logging.info(f"mean nmse {np.mean(nmse_list)}, std nmse {np.std(nmse_list)}")
    logging.info(f"{step} measure total avg after dc")
    logging.info(f"mean psnr {np.mean(psnr_list_adc)}, std psnr {np.std(psnr_list_adc)}")
    logging.info(f"mean ssim {np.mean(ssim_list_adc)}, std ssim {np.std(ssim_list_adc)}")
    logging.info(f"mean mse {np.mean(mse_list_adc)}, std mse {np.std(mse_list_adc)}")
    logging.info(f"mean nmse {np.mean(nmse_list_adc)}, std nmse {np.std(nmse_list_adc)}")





def CMR_sampling_func_Multi_intraining(diffusion_model, ddp_model, output_folder, dataset, step, clip_denoised):
    logging.basicConfig(filename=f'{output_folder}/train_result.log', level=logging.INFO)
    pic_save_path = os.path.join(output_folder,f'Pic_result_{step}/')
    if not os.path.exists(pic_save_path):
        os.makedirs(pic_save_path)
    ddp_model.eval()
    os.makedirs(output_folder, exist_ok=True)
    n_rounds = 1
    print(n_rounds,'nnnn')
    ssim_list, mse_list, nmse_list, psnr_list = [],[],[],[]
    ssim_list_adc, mse_list_adc, nmse_list_adc, psnr_list_adc = [],[],[],[]
    with torch.no_grad():
      with open('/home/zhenlin/norm_test.log', 'r') as file:
          file_content = file.readlines()
      data_idx_list = [i for i in range(len(dataset))]
      data_idx_list = data_idx_list[::10]
      for data_idx in data_idx_list:
          gnd_norm_para = torch.tensor(float(file_content[data_idx].split()[5]))
          und_norm_para = torch.tensor(float(file_content[data_idx].split()[11]))
          sample_index = file_content[data_idx].split()[1]
          one_sample = []
          gt_sample = []
          und_sample = []
          one_sample_afterdc = []
          for idx, slices in enumerate(dataset[data_idx]):
            im_und, im_gnd, mask, sens = slices
            print(im_gnd.shape, im_und.shape)
            gt_mask = im_gnd
            condition_on = im_und
            former_frame_for_feature_extraction = condition_on.to(dist_util.dev())
            x_pred = 0
            pred_cur_afterdc1 = 0
            for round_index in range(n_rounds):
                print(f"Current Round: {round_index+1} / Total Round: {n_rounds}")
                model_kwargs = {
                  "conditioned_image": former_frame_for_feature_extraction}
                x = diffusion_model.ddim_sample_loop(
                  ddp_model,
                  (former_frame_for_feature_extraction.shape[0], gt_mask.shape[1], former_frame_for_feature_extraction.shape[2],
                      former_frame_for_feature_extraction.shape[3]),
                  progress=True,
                  clip_denoised=clip_denoised,
                  model_kwargs=model_kwargs
              )
                pred_cur = x.cpu()
              
                im_gnd_show = im_gnd[:,0,...]+1j*im_gnd[:,1,...]
                plt.imsave(pic_save_path + f'im_gnd_sample{sample_index}_slice{idx}.png',abs(im_gnd_show[0,...]), cmap='gray')
                im_und_show = im_und[:,0,...]+1j*im_und[:,1,...]
                plt.imsave(pic_save_path + f'im_und_sample{sample_index}_slice{idx}.png',abs(im_und_show[0,...]), cmap='gray')

                show_img_before_dc = pred_cur[:,0,...]+1j*pred_cur[:,1,...]
                plt.imsave(pic_save_path + f'before_dc_sample{sample_index}_slice{idx}.png',abs(show_img_before_dc[0,...]), cmap='gray')
                psnr_be = psnr(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_before_dc)) / np.array(abs(show_img_before_dc)).max())
                ssim_be = ssim(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_before_dc)) / np.array(abs(show_img_before_dc)).max())
                print(f'slice {idx} before dc', psnr_be, 'psnr', ssim_be, 'ssim')
                print('pred scale',abs(pred_cur).max(), abs(pred_cur).min(), abs(pred_cur).mean(),'normed',abs(pred_cur*gnd_norm_para).max())
                print('und scale',abs(im_und).max(), abs(im_und).min(), abs(im_und).mean(),'normed',abs(im_und*und_norm_para).max())
                
                pred_cur_dc = pred_cur[:,0:1,...]+1j*pred_cur[:,1:2,...]
                im_und_dc = im_und[:,0:1,...]+1j*im_und[:,1:2,...]
                pred_cur_afterdc = data_consistency_Sens_slice(pred_cur_dc, im_und_dc, mask, np.array(sens))
                pred_cur_afterdc_norm = data_consistency_Sens_slicenorm(pred_cur_dc, im_und_dc, mask, np.array(sens))
                print(pred_cur_afterdc.shape,'afdc shape')
                show_img_after_dc = pred_cur_afterdc[:,0,...]
                show_img_after_dc_norm = pred_cur_afterdc_norm[:,0,...]
                plt.imsave(pic_save_path + f'after_dc_sample{sample_index}_slice{idx}.png',abs(show_img_after_dc[0,...]), cmap='gray')
              
                psnr_af = psnr(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_after_dc)) / np.array(abs(show_img_after_dc)).max())
                ssim_af = ssim(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_after_dc)) / np.array(abs(show_img_after_dc)).max())

                psnr_afn = psnr(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_after_dc_norm)) / np.array(abs(show_img_after_dc_norm)).max())
                ssim_afn = ssim(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_after_dc_norm)) / np.array(abs(show_img_after_dc_norm)).max())
                
                
                print(f' slice {idx} after dc',psnr_af, 'psnr', ssim_af, 'ssim')
                print(f' slice {idx} after dc norm',psnr_afn, 'psnr', ssim_afn, 'ssim')
              
                error = abs(im_gnd_show[0,...])-abs(show_img_before_dc[0,...])
                plt.imsave(pic_save_path + f'error_sample{sample_index}_slice{idx}.png',error, cmap='gray')

                x_pred = x_pred + np.array(pred_cur)
                pred_cur_afterdc1 = pred_cur_afterdc1 + np.array(pred_cur_afterdc)

            x_pred_avg = x_pred/n_rounds
            pred_cur_afterdc_avg = pred_cur_afterdc1/n_rounds

            one_sample.append(x_pred_avg)
            one_sample_afterdc.append(pred_cur_afterdc_avg[:,0,...])
            gt_sample.append(gt_mask)
            und_sample.append(im_und)
          
          #from [s, t, c, h, w] to [t,c,s,h,w] then [t,s,h,w]
          one_sample1 = np.array(one_sample)
          one_sample1 = np.transpose(one_sample1, (1,2,0,3,4))
          one_sample = one_sample1[:,0,...] + 1j*one_sample1[:,1,...]
            
          one_sample_afterdc = np.array(one_sample_afterdc)
          pred_cur_afterdc = np.transpose(one_sample_afterdc, (1,0,2,3))         
          
            
          gt_sample1 = np.array(gt_sample)
          gt_sample1 = np.transpose(gt_sample1, (1,2,0,3,4))
          gt_sample = gt_sample1[:,0,...] + 1j*gt_sample1[:,1,...]

          und_sample1 = np.array(und_sample)
          und_sample1 = np.transpose(und_sample1, (1,2,0,3,4))   
          und_sample = und_sample1[:,0,...] + 1j*und_sample1[:,1,...]
                   
          # print(one_sample1.shape, 'one_sample1', gt_sample1.shape, 'gt_sample1')
          # one_sample_complex = one_sample1[:,0,...] + 1j*one_sample1[:,1,...]
          # gt_sample_complex = gt_sample1[:,0,...] + 1j*gt_sample1[:,1,...]
            
          # pred_cur_afterdc = data_consistency_Sens(one_sample, und_sample, mask_sample, sens_sample1)
          print(one_sample.shape, 'one_sample1', gt_sample.shape, 'gt_sample1', pred_cur_afterdc.shape)
          psnr_ = psnr(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(one_sample)) / np.array(abs(one_sample)).max())
          print('onepsnr', psnr_,)
          psnr_afterdc = psnr(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(pred_cur_afterdc)) / np.array(abs(pred_cur_afterdc)).max())
          # print('onepsnr_afterdc', psnr_afterdc)
        
          mse_ = mse(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(one_sample)) / np.array(abs(one_sample)).max())
          mse_afterdc = mse(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(pred_cur_afterdc)) / np.array(abs(pred_cur_afterdc)).max())          
            
          nmse_ = nmse(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(one_sample)) / np.array(abs(one_sample)).max())
          nmse_afterdc = nmse(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(pred_cur_afterdc)) / np.array(abs(pred_cur_afterdc)).max())
            
          ssim_ = ssim(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(one_sample)) / np.array(abs(one_sample)).max())
          ssim_afterdc = ssim(np.array(abs(gt_sample))/np.array(abs(gt_sample)).max(), np.array(abs(pred_cur_afterdc)) / np.array(abs(pred_cur_afterdc)).max())
          
            
          print('onessim',ssim_,)
          print('onessim after dc',ssim_afterdc)
            
          psnr_list.append(psnr_)
          mse_list.append(mse_)
          nmse_list.append(nmse_)
          ssim_list.append(ssim_)
          psnr_list_adc.append(psnr_afterdc)
          mse_list_adc.append(mse_afterdc)
          nmse_list_adc.append(nmse_afterdc)
          ssim_list_adc.append(ssim_afterdc)
        
          logging.info(
              f"{sample_index} n round psnr {psnr_list[-1]}, ssim {ssim_list[-1]}, mse {mse_list[-1]}, nmse {nmse_list[-1]}")
          logging.info(
              f"{sample_index} n round after dc psnr {psnr_list_adc[-1]}, ssim {ssim_list_adc[-1]}, mse {mse_list_adc[-1]}, nmse {nmse_list_adc[-1]}")
        
          print("current avg PSNR:", np.mean(psnr_list), 'current avg after dc PSNR:',np.mean(psnr_list_adc))
          print("current avg SSIM:", np.mean(ssim_list), 'current avg after dc SSIM:',np.mean(ssim_list_adc))
          print("current avg NMSE:", np.mean(nmse_list), 'current avg after dc NMSE:',np.mean(nmse_list_adc))

    my_length = len(psnr_list)
    print('psnr_lenth', my_length)
    logging.info(f" {step} measure total avg")
    logging.info(f"mean psnr {np.mean(psnr_list)}, std psnr {np.std(psnr_list)}")
    logging.info(f"mean ssim {np.mean(ssim_list)}, std ssim {np.std(ssim_list)}")
    logging.info(f"mean mse {np.mean(mse_list)}, std mse {np.std(mse_list)}")
    logging.info(f"mean nmse {np.mean(nmse_list)}, std nmse {np.std(nmse_list)}")
    logging.info(f" {step} measure total avg after dc")
    logging.info(f"mean psnr {np.mean(psnr_list_adc)}, std psnr {np.std(psnr_list_adc)}")
    logging.info(f"mean ssim {np.mean(ssim_list_adc)}, std ssim {np.std(ssim_list_adc)}")
    logging.info(f"mean mse {np.mean(mse_list_adc)}, std mse {np.std(mse_list_adc)}")
    logging.info(f"mean nmse {np.mean(nmse_list_adc)}, std nmse {np.std(nmse_list_adc)}")

def ZF_metrics(diffusion_model, ddp_model, output_folder, dataset, logger, clip_denoised, pic_save_path):
    if not os.path.exists(pic_save_path):
        os.makedirs(pic_save_path)
    ddp_model.eval()
    os.makedirs(output_folder, exist_ok=True)
    n_rounds = 1
    print(n_rounds,'nnnn')
    ssim_list, mse_list, nmse_list, psnr_list = [],[],[],[]
    with torch.no_grad():
      with open('/home/zhenlin/norm_test.log', 'r') as file:
          file_content = file.readlines()
      data_idx_list = [i for i in range(len(dataset))]
      # data_idx_list = [0]
      for data_idx in data_idx_list:
          gnd_norm_para = torch.tensor(float(file_content[data_idx].split()[5]))
          und_norm_para = torch.tensor(float(file_content[data_idx].split()[11]))
          sample_index = file_content[data_idx].split()[1]
          one_sample = []
          gt_sample = []
          one_sample_afterdc = []
          und_sample = []
          for idx, slices in enumerate(dataset[data_idx]):
            im_und, im_gnd, mask = slices
            print(im_gnd.shape, im_und.shape)
            gt_mask = im_gnd
            condition_on = im_und
            for round_index in range(n_rounds):
              
              im_gnd_show = im_gnd[:,0,...]+1j*im_gnd[:,1,...]
              plt.imsave(pic_save_path + f'im_gnd_sample{sample_index}_slice{idx}.png',abs(im_gnd_show[0,...]), cmap='gray')
              im_und_show = im_und[:,0,...]+1j*im_und[:,1,...]
              plt.imsave(pic_save_path + f'im_und_sample{sample_index}_slice{idx}.png',abs(im_und_show[0,...]), cmap='gray')
              
              psnr_base = psnr(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(im_und_show)) / np.array(abs(im_und_show)).max())
              ssim_base = ssim(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(im_und_show)) / np.array(abs(im_und_show)).max())
              print('und scale',abs(im_und).max(), abs(im_und).min(), abs(im_und).mean(),'normed',abs(im_und*und_norm_para).max())

                
            gt_sample.append(gt_mask)
            und_sample.append(im_und)
            
          gt_sample1 = np.array(gt_sample)
          gt_sample1 = np.transpose(gt_sample1, (1,2,0,3,4))
          
          und_sample1 = np.array(und_sample)
          und_sample1 = np.transpose(und_sample1, (1,2,0,3,4))

            
          print( gt_sample1.shape, 'gt_sample1')

          gt_sample_complex = gt_sample1[:,0,...] + 1j*gt_sample1[:,1,...]
          und_sample_complex = und_sample1[:,0,...] + 1j*und_sample1[:,1,...]

          psnr_ = psnr(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(und_sample_complex)) / np.array(abs(und_sample_complex)).max())
          print('onepsnr', psnr_,)
          mse_ = mse(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(und_sample_complex)) / np.array(abs(und_sample_complex)).max())
          nmse_ = nmse(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(und_sample_complex)) / np.array(abs(und_sample_complex)).max())    
          ssim_ = ssim(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(und_sample_complex)) / np.array(abs(und_sample_complex)).max())
            
          print('onessim',ssim_,)
            
          psnr_list.append(psnr_)
          mse_list.append(mse_)
          nmse_list.append(nmse_)
          ssim_list.append(ssim_)
        
        
          logger.info(
              f"{sample_index} n round psnr {psnr_list[-1]}, ssim {ssim_list[-1]}, mse {mse_list[-1]}, nmse {nmse_list[-1]}")

        
          print("current ZF avg PSNR:", np.mean(psnr_list))
          print("current ZF avg SSIM:", np.mean(ssim_list))
          print("current ZF avg NMSE:", np.mean(nmse_list))

    my_length = len(psnr_list)
    print('psnr_lenth', my_length)
    logger.info("measure total avg ZF")
    logger.info(f"mean psnr {np.mean(psnr_list)}, std psnr {np.std(psnr_list)}")
    logger.info(f"mean ssim {np.mean(ssim_list)}, std ssim {np.std(ssim_list)}")
    logger.info(f"mean mse {np.mean(mse_list)}, std mse {np.std(mse_list)}")
    logger.info(f"mean nmse {np.mean(nmse_list)}, std nmse {np.std(nmse_list)}")


def ZF_metrics_multi(diffusion_model, ddp_model, output_folder, dataset, logger, clip_denoised, pic_save_path):
    if not os.path.exists(pic_save_path):
        os.makedirs(pic_save_path)
    ddp_model.eval()
    os.makedirs(output_folder, exist_ok=True)
    n_rounds = 1
    ssim_list, mse_list, nmse_list, psnr_list = [],[],[],[]
    ssim_list_adc, mse_list_adc, nmse_list_adc, psnr_list_adc = [],[],[],[]
    logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  
    filename=output_folder+'/combined result.log',  
    filemode='w'  # 文件模式，'a' 代表追加模式，'w' 代表覆盖模式
)
    logger_com = logging.getLogger(__name__)
    with torch.no_grad():
        data_idx_list = [i for i in range(len(dataset))]

        ssim_coil, mse_coil, nmse_coil, psnr_coil = [],[],[],[]
        gnd_eachcoil = []
        und_eachcoil = []
        for data_idx in data_idx_list:
            gt_coil = []
            und_coil = []
            for idx, slices in enumerate(dataset[data_idx]):
                sample_index = dataset.all_patient[data_idx]
                im_und, im_gnd, mask = slices
                print(im_gnd.shape, im_und.shape)
                gt_mask = im_gnd
                for round_index in range(n_rounds):
                    print(f"Current Round: {round_index+1} / Total Round: {n_rounds}")
                   
              
                    im_gnd_show = im_gnd[:,0,...]+1j*im_gnd[:,1,...]
                    im_und_show = im_und[:,0,...]+1j*im_und[:,1,...]
                    psnr_base = psnr(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(im_und_show)) / np.array(abs(im_und_show)).max())
                    ssim_base = ssim(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(im_und_show)) / np.array(abs(im_und_show)).max())
                    print(f'slice {idx} base', psnr_base, 'psnr', ssim_base, 'ssim')
               
                    
                gt_coil.append(gt_mask)
                und_coil.append(im_und)
                
 
            gt_coil1 = np.array(gt_coil)
            gt_coil1 = np.transpose(gt_coil1, (1,2,0,3,4))
            und_coil1 = np.array(und_coil)
            und_coil1 = np.transpose(und_coil1, (1,2,0,3,4))
            
            gt_sample_complex = gt_coil1[:,0,...] + 1j*gt_coil1[:,1,...]
            und_sample_complex = und_coil1[:,0,...] + 1j*und_coil1[:,1,...]

            gnd_eachcoil.append(gt_sample_complex)
            und_eachcoil.append(und_sample_complex)

            print(gt_sample_complex.shape, 'gt_sample1')
            psnr_ = psnr(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(und_sample_complex)) / np.array(abs(und_sample_complex)).max())
            psnr_coil.append(psnr_)
            print('onecoil', psnr_,)
        
            # mse_ = mse(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(und_sample_complex)) / np.array(abs(und_sample_complex)).max())
            # mse_coil.append(mse_)

            # nmse_ = nmse(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(und_sample_complex)) / np.array(abs(und_sample_complex)).max())
            # nmse_coil.append(nmse_)
           
            # ssim_ = ssim(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(und_sample_complex)) / np.array(abs(und_sample_complex)).max())
            # ssim_coil.append(ssim_)
          
            # logger.info(
            #   f"{sample_index} coil{(data_idx+1)%10} psnr {psnr_coil[-1]}, ssim {ssim_coil[-1]}, mse {mse_coil[-1]}, nmse {nmse_coil[-1]}")
            print(len(gnd_eachcoil), len(und_eachcoil))
            if len(gnd_eachcoil) == 10:
                combined_gnd = coil_combination_rss(gnd_eachcoil)
                combined_und = coil_combination_rss(und_eachcoil)
                
                psnr_value, ssim_value, nmse_value, mse_value = cal_four_metrics(combined_gnd, combined_und)

                logger.info(
                 f"{sample_index} combined psnr {psnr_value}, ssim {ssim_value}, nmse {nmse_value}, mse {mse_value}")
                logger.info('==============================================NEXT SAMPLE===============================================================')
                psnr_list.append(psnr_value)
                mse_list.append(mse_value)
                nmse_list.append(nmse_value)
                ssim_list.append(ssim_value)
                logger_com.info(f'PSNR list {psnr_list}')
                logger_com.info(f'MSE list {mse_list}')
                logger_com.info(f'SSIM list {ssim_list}')
                logger_com.info(f'NMSE list {nmse_list}')

                gnd_eachcoil = []
                und_eachcoil = []

                print("current avg PSNR:", np.mean(psnr_list))
                print("current avg SSIM:", np.mean(ssim_list))
                print("current avg NMSE:", np.mean(nmse_list))

    my_length = len(psnr_list)
    print('psnr_lenth', my_length)
    logger.info("=====================measure total avg MULTI================================")
    logger.info(f"mean psnr {np.mean(psnr_list)}, std psnr {np.std(psnr_list)}")
    logger.info(f"mean ssim {np.mean(ssim_list)}, std ssim {np.std(ssim_list)}")
    logger.info(f"mean mse {np.mean(mse_list)}, std mse {np.std(mse_list)}")
    logger.info(f"mean nmse {np.mean(nmse_list)}, std nmse {np.std(nmse_list)}")
