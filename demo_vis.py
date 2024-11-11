import random
from data_loader_Cmask import DataLoaderForVis, TestCmaskNormMultiOneCoil, TrainCmaskNormMultiOneCoil
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
# test_data_vis4 = DataLoaderForVis(4, 'sax','01')
# test_data_vis8 = DataLoaderForVis(8, 'sax','01')
# test_data_vis10 = DataLoaderForVis(10, 'sax','01')

# for im_und_out4, im_gnd_out4, mask4 in test_data_vis4[0]:
#             im_gnd_out4 = im_gnd_out4[:,0,...]+im_gnd_out4[:,1,...]*1j
#             im_und_out4 = im_und_out4[:,0,...]+im_und_out4[:,1,...]*1j
#             im_gnd_out4 = abs(im_gnd_out4[0])
#             im_und_out4 = abs(im_und_out4[0]) 
#             print(im_und_out4.shape)
#             # plt.imsave('gnd_full.png',im_gnd_out4, cmap='gray')  
#             mask4_show =  mask4[0,0]
#             break
        
# for im_und_out8, im_gnd_out8,mask8 in test_data_vis8[0]:
#             im_gnd_out8 = im_gnd_out8[:,0,...]+im_gnd_out8[:,1,...]*1j
#             im_und_out8 = im_und_out8[:,0,...]+im_und_out8[:,1,...]*1j
#             im_gnd_out8 = abs(im_gnd_out8[0])
#             im_und_out8 = abs(im_und_out8[0])    
#             mask8_show =  mask8[0,0]
#             break
# for im_und_out10, im_gnd_out10,mask10 in test_data_vis10[0]:
#             im_gnd_out10 = im_gnd_out10[:,0,...]+im_gnd_out10[:,1,...]*1j
#             im_und_out10 = im_und_out10[:,0,...]+im_und_out10[:,1,...]*1j
#             im_gnd_out10 = abs(im_gnd_out10[0])
#             im_und_out10 = abs(im_und_out10[0])    
#             mask10_show =  mask10[0,0]
#             break
                    


# fig = plt.figure(figsize=(45, 30))
# gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.05, hspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)

#             # 第一行子图
# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[0, 1])
# ax3 = fig.add_subplot(gs[0, 2])
# ax1.imshow(im_gnd_out4, cmap='gray')
# ax1.set_title('Full sampled', fontsize=48)
# ax1.axis('off')
# ax1.annotate('AccFactor 4', xy=(-0.05, 0.5), xycoords='axes fraction', fontsize=48, rotation='vertical', va='center')
# ax2.imshow(mask4_show, cmap='gray')
# ax2.set_title('Mask', fontsize=48)
# ax2.axis('off')
# ax3.set_title('Under sampled', fontsize=48)
# ax3.imshow(im_und_out4, cmap='gray')
# ax3.axis('off')

# # 第二行子图
# ax4 = fig.add_subplot(gs[1, 0])
# ax4.imshow(im_gnd_out8, cmap='gray')
# ax4.axis('off')
# ax5 = fig.add_subplot(gs[1, 1])
# ax6 = fig.add_subplot(gs[1, 2])
# ax5.imshow(mask8_show, cmap='gray')
# ax4.annotate('AccFactor 8', xy=(-0.05, 0.5), xycoords='axes fraction', fontsize=48, rotation='vertical', va='center')
# ax5.axis('off')
# ax6.imshow(im_und_out8, cmap='gray')
# ax6.axis('off')

#  # 第三行子图
# ax7 = fig.add_subplot(gs[2, 0])
# ax8 = fig.add_subplot(gs[2, 1])
# ax9 = fig.add_subplot(gs[2, 2])
# ax7.imshow(im_gnd_out10, cmap='gray')
# ax7.axis('off')
# ax8.imshow(mask10_show, cmap='gray')
# ax7.annotate('AccFactor 10', xy=(-0.05, 0.5), xycoords='axes fraction', fontsize=48, rotation='vertical', va='center')
# ax8.axis('off')
# ax9.imshow(im_und_out10, cmap='gray')
# ax9.axis('off')

# plt.savefig(f'multiacc_show.png', bbox_inches='tight')
# plt.show()

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

# Testdata = TrainCmaskNormMultiOneCoil(8, 'sax', norm = '01', patch=256)
# coil_list = []
# for i in range(10):
#     samples = Testdata[i]
#     slices = []
#     for im_und_out, im_gnd_out in samples:
#         im_gnd_out = im_gnd_out[:,0:1,...]+im_gnd_out[:,1:2,...]*1j
#         slices.append(im_gnd_out)
#     coil_list.append(slices[0])
#     plt.imshow(abs(slices[0][0,0,...]), cmap='gray')
#     plt.imsave(f"/home/zhenlin/DiffMamba/gnd_coil{i}.png",abs(slices[0][0,0,...]), cmap='gray')
#     plt.show()
# combined = coil_combination_rss(coil_list)
# plt.imsave(f"/home/zhenlin/DiffMamba/combined.png",abs(combined[0,0,...]), cmap='gray')

# print(len(Testdata))

from PIL import Image
import numpy as np
import cv2
sample = '04'
slices = '3'
image_gnd = np.array(Image.open(f'/home/zhenlin/DiffMamba/Pic_Result/SSMVideo_c8/model_238800/im_gnd_sampleP0{sample}_slice{slices}.png'))

image_und = np.array(Image.open(f'/home/zhenlin/DiffMamba/Pic_Result/SSMVideo_c8/model_238800/im_und_sampleP0{sample}_slice{slices}.png'))

image_pred_v5 = np.array(Image.open(f"/home/zhenlin/DiffMamba/Pic_Result/v5_cmask8/model_250740_v5_c8/after_dc_sampleP0{sample}_slice{slices}.png"))

image_pred_ssm = np.array(Image.open(f"/home/zhenlin/DiffMamba/Pic_Result/SSMVideo_c8/model_238800/after_dc_sampleP0{sample}_slice{slices}.png"))

image_pred_attn = np.array(Image.open(f"/home/zhenlin/DiffCMR_Multich2/Pic_results/CDiff_Full_Temattn/08/model_274620/after_dc_sampleP0{sample}_slice{slices}.png"))

image_pred_cmr = np.array(Image.open(f"/home/zhenlin/DiffCMR_Multich2/Pic_results/CDiff_base/08/model_465660/after_dc_sampleP0{sample}_slice{slices}.png"))


diff_v5 = cv2.absdiff(image_gnd, image_pred_v5)
# cv2.imwrite('/home/zhenlin/DiffMamba/Pic_Result/ZF_results/08/diff.png', diff_base[...,0])
plt.imsave("/home/zhenlin/DiffMamba/Pic_Result_Multi/diff_v5_single.png",diff_v5[...,0]/255, vmin=0, vmax=0.15)
plt.imshow(diff_v5[...,0]/255, vmin=0, vmax=0.15)
# plt.colorbar()
plt.show()


diff_ssm = cv2.absdiff(image_gnd, image_pred_ssm)
# cv2.imwrite('/home/zhenlin/DiffMamba/Pic_Result/ZF_results/08/diff.png', diff_base[...,0])
plt.imsave("/home/zhenlin/DiffMamba/Pic_Result_Multi/diff_ssmvideo_single.png",diff_ssm[...,0]/255, vmin=0, vmax=0.15)
plt.imshow(diff_ssm[...,0]/255, vmin=0, vmax=0.15)
# plt.colorbar()
plt.show()

diff_attn = cv2.absdiff(image_gnd, image_pred_attn)
# cv2.imwrite('/home/zhenlin/DiffMamba/Pic_Result/ZF_results/08/diff.png', diff_base[...,0])
plt.imsave("/home/zhenlin/DiffMamba/Pic_Result_Multi/diff_attn_single.png",diff_attn[...,0]/255, vmin=0, vmax=0.15)
plt.imshow(diff_attn[...,0]/255, vmin=0, vmax=0.15)
# plt.colorbar()
plt.show()

diff_cmr = cv2.absdiff(image_gnd, image_pred_cmr)
# cv2.imwrite('/home/zhenlin/DiffMamba/Pic_Result/ZF_results/08/diff.png', diff_base[...,0])
plt.imsave("/home/zhenlin/DiffMamba/Pic_Result_Multi/diff_cmr_single.png",diff_cmr[...,0]/255, vmin=0, vmax=0.15)
plt.imshow(diff_cmr[...,0]/255, vmin=0, vmax=0.15)
# plt.colorbar()
plt.show()

diff_zf = cv2.absdiff(image_gnd, image_und)
# cv2.imwrite('/home/zhenlin/DiffMamba/Pic_Result/ZF_results/08/diff.png', diff_base[...,0])
plt.imsave("/home/zhenlin/DiffMamba/Pic_Result_Multi/diff_zf_single.png",diff_zf[...,0]/255, vmin=0, vmax=0.15)
plt.imshow(diff_zf[...,0]/255, vmin=0, vmax=0.15)
plt.colorbar()
plt.show()

image_gnd = np.array(Image.open(f'/home/zhenlin/DiffMamba/Pic_Result_Multi/v5_multi/cmask8_2th/gnd_combined_sampleP0{sample}_slice{slices}.png'))

image_und = np.array(Image.open(f'/home/zhenlin/DiffMamba/Pic_Result_Multi/v5_multi/cmask8_2th/und_combined_sampleP0{sample}_slice{slices}.png'))

image_pred_v5 = np.array(Image.open(f"/home/zhenlin/DiffMamba/Pic_Result_Multi/v5_multi/cmask8_2th/after_dc_combined_sampleP0{sample}_slice{slices}.png"))

image_pred_ssm = np.array(Image.open(f"/home/zhenlin/DiffMamba/Pic_Result_Multi/SSMVideo/cmask8_2th/after_dc_combined_sampleP0{sample}_slice{slices}.png"))

image_pred_cmr = np.array(Image.open(f"/home/zhenlin/DiffMamba/Pic_Result_Multi/DiffCMR_multi/cmask8_2th/after_dc_combined_sampleP0{sample}_slice{slices}.png"))


diff_v5 = cv2.absdiff(image_gnd, image_pred_v5)
# cv2.imwrite('/home/zhenlin/DiffMamba/Pic_Result/ZF_results/08/diff.png', diff_base[...,0])
plt.imsave("/home/zhenlin/DiffMamba/Pic_Result_Multi/diff_v5.png",diff_v5[...,0]/255, vmin=0, vmax=0.15)
plt.imshow(diff_v5[...,0]/255, vmin=0, vmax=0.15)
# plt.colorbar()
plt.show()

diff_ssm = cv2.absdiff(image_gnd, image_pred_ssm)
# cv2.imwrite('/home/zhenlin/DiffMamba/Pic_Result/ZF_results/08/diff.png', diff_base[...,0])
plt.imsave("/home/zhenlin/DiffMamba/Pic_Result_Multi/diff_ssmvideo.png",diff_ssm[...,0]/255, vmin=0, vmax=0.15)
plt.imshow(diff_ssm[...,0]/255, vmin=0, vmax=0.15)
# plt.colorbar()
plt.show()


diff_cmr = cv2.absdiff(image_gnd, image_pred_cmr)
# cv2.imwrite('/home/zhenlin/DiffMamba/Pic_Result/ZF_results/08/diff.png', diff_base[...,0])
plt.imsave("/home/zhenlin/DiffMamba/Pic_Result_Multi/diff_cmr.png",diff_cmr[...,0]/255, vmin=0, vmax=0.15)
plt.imshow(diff_cmr[...,0]/255, vmin=0, vmax=0.15)
# plt.colorbar()
plt.show()

diff_zf = cv2.absdiff(image_gnd, image_und)
# cv2.imwrite('/home/zhenlin/DiffMamba/Pic_Result/ZF_results/08/diff.png', diff_base[...,0])
plt.imsave("/home/zhenlin/DiffMamba/Pic_Result_Multi/diff_zf.png",diff_zf[...,0]/255, vmin=0, vmax=0.15)
plt.imshow(diff_zf[...,0]/255, vmin=0, vmax=0.15)
plt.colorbar()
plt.show()
# diff_gnd = cv2.absdiff(image_gnd, image_gnd)
# # cv2.imwrite('/home/zhenlin/DiffMamba/Pic_Result/ZF_results/08/diff.png', diff_base[...,0])
# plt.imsave("/home/zhenlin/DiffMamba/Pic_Result/v5_cmask8/diff_gnd.png",diff_gnd[...,0]/255, vmin=0, vmax=0.15)
# plt.imshow(diff_gnd[...,0]/255, vmin=0, vmax=0.15)
# plt.colorbar()
# plt.show()

# fig = plt.figure(figsize=(4, 9))
# gs = gridspec.GridSpec(6, 2, figure=fig, wspace=0.1, hspace=0.1, left=0.1, right=0.95, top=0.95, bottom=0.05)
# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[0, 1])

# ax1.imshow(image_und, cmap='gray')
# # ax1.set_title('Full sampled', fontsize=48)
# ax1.axis('off')
# ax1.annotate('AccFactor 8', xy=(-0.05, 0.5), xycoords='axes fraction', fontsize=5, rotation='vertical', va='center')


# image_error_zf = np.array(Image.open("/home/zhenlin/DiffMamba/Pic_Result/ZF_results/08/diff_zf.png"))
# ax2.imshow(image_error_zf)
# ax2.axis('off')
# ax3 = fig.add_subplot(gs[1, 0])
# ax3.imshow(image_pred_v5, cmap='gray' )
# ax3.axis('off')
# ax4 = fig.add_subplot(gs[1, 1])
# image_error_v5 = np.array(Image.open('/home/zhenlin/DiffMamba/Pic_Result/v5_cmask8/diff_v5.png'))
# ax4.imshow(image_error_v5)
# ax4.axis('off')
# plt.savefig(f'results.png', bbox_inches='tight')



# ax5 = fig.add_subplot(gs[2, 0])
# ax5.axis('off')
# ax6 = fig.add_subplot(gs[2, 1])
# ax6.axis('off')

# ax7 = fig.add_subplot(gs[3, 0])
# ax8 = fig.add_subplot(gs[3, 1])

# ax9 = fig.add_subplot(gs[4, 0])
# ax10 = fig.add_subplot(gs[4, 1])
# # plt.imshow(diff_base, vmin=0, vmax=0.1)
# # plt.colorbar()
# # plt.show()
# ax11 = fig.add_subplot(gs[5, 0])
# ax12 = fig.add_subplot(gs[5, 1])

