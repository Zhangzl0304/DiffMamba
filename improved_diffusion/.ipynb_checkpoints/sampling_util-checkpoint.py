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

from . import dist_util
from .metrics import FBound_metric, WCov_metric
# from datasets.monu import MonuDataset
from .utils import set_random_seed_for_iterations
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
    out = ifft2c((1 - mask) * fft2c(im_pred) + mask * fft2c(im_und))
    return out



def CMR_sampling_major_vote_func_with_dc(diffusion_model, ddp_model, output_folder, dataset, logger, clip_denoised, dc_step, pic_save_path):
    if not os.path.exists(pic_save_path):
        os.makedirs(pic_save_path)
    ddp_model.eval()
    os.makedirs(output_folder, exist_ok=True)
    n_rounds = 1
    print(n_rounds,'nnnn')

    ssim_list, mse_list, nmse_list, psnr_list = [],[],[],[]
    with torch.no_grad():
      with open('/rds/general/user/zz8523/home/DTiffCMR_Multich/norm_test.log', 'r') as file:
          file_content = file.readlines()
      for data_idx in range(len(dataset)):
          gnd_norm_para = torch.tensor(float(file_content[data_idx].split()[5]))
          und_norm_para = torch.tensor(float(file_content[data_idx].split()[11]))
          sample_index = file_content[data_idx].split()[1]
          one_sample = []
          gt_sample = []
          und_sample = []
          print("Sample idx", sample_index)
          for idx, slices in enumerate(dataset[data_idx]):
            
            im_und, im_gnd, mask = slices
            print('im_gnd, im_und shape',im_gnd.shape, im_und.shape)
            condition_on = im_und
            former_frame_for_feature_extraction = condition_on.to(dist_util.dev())
            x_pred = 0
            for round_index in range(n_rounds):
              print(f"Current Round: {round_index+1} / Total Round: {n_rounds}")
              model_kwargs = {
                "conditioned_image": former_frame_for_feature_extraction}
              x = diffusion_model.ddim_sample_loop(
                ddp_model,
                (former_frame_for_feature_extraction.shape[0], im_gnd.shape[1], former_frame_for_feature_extraction.shape[2],
                    former_frame_for_feature_extraction.shape[3]),
                progress=True,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs
                  )
              pred_cur = x.cpu()
              
              im_gnd_show = im_gnd[:,0,...]+1j*im_gnd[:,1,...]
              plt.imsave(pic_save_path + f'im_gnd_sample{sample_index}_slice{idx}.png',abs(im_gnd_show[0,...]), cmap='gray')
              img_pred = pred_cur[:,0,...]+1j*pred_cur[:,1,...]
              plt.imsave(pic_save_path + f'im_pred_sample{sample_index}_slice{idx}.png',abs(img_pred[0,...])/abs(img_pred[0,...]).max(), cmap='gray')
              img_input = im_und[:,0,...]+1j*im_und[:,1,...]
              plt.imsave(pic_save_path + f'im_input_sample{sample_index}_slice{idx}.png',abs(img_input[0,...]), cmap='gray')
                        

              psnr_und = psnr(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(img_input)) / np.array(abs(img_input)).max())
              ssim_und = ssim(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(img_input)) / np.array(abs(img_input)).max())
              print("slice und:",idx, 'psnr:', psnr_und, 'ssim:', ssim_und)
              psnr_be = psnr(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(img_pred)) / np.array(abs(img_pred)).max())
              ssim_be = ssim(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(img_pred)) / np.array(abs(img_pred)).max())
              print("slice pred:",idx, 'psnr before dc:', psnr_be, 'ssim before dc:', ssim_be)
#               pred_cur = data_consistency(pred_cur*gnd_norm_para, im_und*und_norm_para, mask)
              show_img_after_dc = pred_cur[:,0,...]+1j*pred_cur[:,1,...]
#               plt.imsave(pic_save_path + f'after_dc_sample{sample_index}_slice{idx}.png',abs(show_img_after_dc[0,...]), cmap='gray')
              psnr_af = psnr(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_after_dc)) / np.array(abs(show_img_after_dc)).max())
              ssim_af = ssim(np.array(abs(im_gnd_show))/np.array(abs(im_gnd_show)).max(), np.array(abs(show_img_after_dc)) / np.array(abs(show_img_after_dc)).max())
              print('psnr after dc',psnr_af,'ssim after dc', ssim_af)
              
            
              error = abs(abs(im_gnd_show[0,...])/abs(im_gnd_show[0,...]).max()-abs(img_pred[0,...])/abs(img_pred[0,...]).max())
              plt.clf()
              plt.imshow(error, vmin=0, vmax=0.1)
              plt.colorbar()
              plt.savefig(pic_save_path + f'error_sample{sample_index}_slice{idx}.png', bbox_inches='tight')
#               plt.imsave(pic_save_path + f'error_sample{sample_index}_slice{idx}.png',error, cmap='gray')
              x_pred = x_pred + np.array(pred_cur)
             
            x_pred_avg = x_pred/n_rounds
#             show_img = x_pred_avg[:,0,...]+1j*x_pred_avg[:,1,...]
#             show_gt = im_gnd[:,0,...]+1j*im_gnd[:,1,...]
#             psnr_slice = psnr(np.array(abs(show_gt))/np.array(abs(show_gt)).max(), np.array(abs(show_img)) / np.array(abs(show_img)).max())
#             print('psnr one slice n_rounds',psnr_slice)
#             psnr_be = psnr(np.array(abs(show_gt))/np.array(abs(show_gt)).max(), np.array(abs(show_img)) / np.array(abs(show_img)).max())
#             ssim_be = ssim(np.array(abs(show_gt[0,...]))/np.array(abs(show_gt[0,...])).max(), np.array(abs(show_img[0,...])) / np.array(abs(img_pred[0,...])).max())
#             print("slice pred:",idx, 'psnr:', psnr_be, 'ssim:', ssim_be)
#             psnr_und = psnr(np.array(abs(show_gt))/np.array(abs(show_gt)).max(), np.array(abs(img_input)) / np.array(abs(img_input)).max())
#             ssim_und = ssim(np.array(abs(show_gt[0,...]))/np.array(abs(show_gt[0,...])).max(), np.array(abs(img_input[0,...])) / np.array(abs(img_input[0,...])).max())
#             print("slice und:",idx, 'psnr:', psnr_und, 'ssim:', ssim_und)

            one_sample.append(x_pred_avg)
            gt_sample.append(im_gnd)
            und_sample.append(im_und)
          one_sample1 = np.array(one_sample)
          one_sample1 = np.transpose(one_sample1, (1,2,0,3,4))
          gt_sample1 = np.array(gt_sample)
          gt_sample1 = np.transpose(gt_sample1, (1,2,0,3,4))
          und_sample1 = np.array(und_sample)
          und_sample1 = np.transpose(und_sample1, (1,2,0,3,4))
          print(one_sample1.shape, 'one_sample1', gt_sample1.shape, 'gt_sample1')
          one_sample_complex = one_sample1[:,0,...] + 1j*one_sample1[:,1,...]
          gt_sample_complex = gt_sample1[:,0,...] + 1j*gt_sample1[:,1,...]
          und_sample_complex = und_sample1[:,0,...] + 1j*und_sample1[:,1,...]
          print(one_sample_complex.shape, 'one_sample1', gt_sample_complex.shape, 'gt_sample1')
          psnr_ = psnr(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_complex)) / np.array(abs(one_sample_complex)).max())
          print(psnr_,'onepsnr')
          psnr_base = psnr(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(und_sample_complex)) / np.array(abs(und_sample_complex)).max())
          print(psnr_base,'base psnr')
          mse_ = mse(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_complex)) / np.array(abs(one_sample_complex)).max())
          nmse_ = nmse(np.array(abs(gt_sample_complex))/np.array(abs(gt_sample_complex)).max(), np.array(abs(one_sample_complex)) / np.array(abs(one_sample_complex)).max())
          ssim_ = ssim(np.array(abs(gt_sample_complex[0,...]))/np.array(abs(gt_sample_complex[0,...])).max(), np.array(abs(one_sample_complex[0,...])) / np.array(abs(one_sample_complex[0,...])).max())
          print(ssim_,'onessim')
          psnr_list.append(psnr_)
          mse_list.append(mse_)
          nmse_list.append(nmse_)
          ssim_list.append(ssim_)
          logger.info(
              f"{sample_index} n round psnr {psnr_list[-1]}, ssim {ssim_list[-1]}, mse {mse_list[-1]}, nmse {nmse_list[-1]}")
          print("current avg PSNR:", np.mean(psnr_list))
          print("current avg SSIM:", np.mean(ssim_list))
          print("current avg NMSE:", np.mean(nmse_list))

    my_length = len(psnr_list)
    print('psnr_lenth', my_length)
    logger.info("measure total avg")
    logger.info(f"mean psnr {np.mean(psnr_list)}, std psnr {np.std(psnr_list)}")
    logger.info(f"mean ssim {np.mean(ssim_list)}, std ssim {np.std(ssim_list)}")
    logger.info(f"mean mse {np.mean(mse_list)}, std mse {np.std(mse_list)}")
    logger.info(f"mean nmse {np.mean(nmse_list)}, std nmse {np.std(nmse_list)}")


