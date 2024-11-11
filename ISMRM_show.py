import random
from data_loader_Cmask import DataLoaderForVis, TestCmaskNormMultiOneCoil, TrainCmaskNormMultiOneCoil
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.ndimage import zoom
import os
from PIL import Image
import cv2


pic_save_path= "/home/zhenlin/DiffMamba/ISMRM/Pic/P015/Tcmr/"
os.makedirs(pic_save_path, exist_ok=True)

zoom_factors = (3, 1)  
# /home/zhenlin/DiffMamba/ISMRM/P001_result.npz
# '/bigdata/zhenlin/DiffMamba/SingleCoil/v5_cmask8/P001_results.npz'
loaded = np.load('/bigdata/zhenlin/DiffTCMR_multi/C8/P015_result.npz')
gnd, und, pred = abs(loaded['gnd']), abs(loaded['und']), abs(loaded['pred'])
gnd = gnd/gnd.max()
und = und/und.max()
pred = pred/pred.max()
print(pred)

sample_methods = 'Tcmr'
slice = 6
t,s,h,w = gnd.shape
print(gnd.shape, und.shape, pred.shape)
plt.imshow(abs(gnd[0,slice,...]), cmap='gray')
plt.show()
plt.imshow(abs(und[0,slice,...]), cmap='gray')
plt.show()
plt.imshow(abs(pred[0,slice,...]), cmap='gray')
plt.show()
print(abs(gnd-und).mean())
print(abs(gnd-pred).mean())

# draw temporal pic

wide_image_pred = zoom(abs(pred[:,slice,:,int(w/2)]), zoom_factors, order=2)
wide_image_pred = wide_image_pred.transpose(1,0).astype(np.float32)
plt.imshow(wide_image_pred, cmap='gray')
plt.show()

wide_image_gnd = zoom(abs(gnd[:,slice,:,int(w/2)]), zoom_factors, order=2)
wide_image_gnd = wide_image_gnd.transpose(1,0).astype(np.float32)
plt.imshow(wide_image_gnd, cmap='gray')
plt.show()

wide_image_und = zoom(abs(und[:,slice,:,int(w/2)]), zoom_factors, order=2)
wide_image_und = wide_image_und.transpose(1,0).astype(np.float32)
plt.imshow(wide_image_und, cmap='gray')
plt.show()
plt.imsave(pic_save_path+ f"img_pred_{sample_methods}_t.png",wide_image_pred, cmap='gray')
plt.imsave(pic_save_path+ f"img_gnd_t.png",wide_image_gnd, cmap='gray')
plt.imsave(pic_save_path+ f"img_und_t.png",wide_image_und, cmap='gray')

# draw temporal diff



diff_und = cv2.absdiff(wide_image_gnd, wide_image_und)
print(diff_und.max(), diff_und.mean())
plt.imshow(diff_und, vmin=0, vmax=0.1)
plt.imsave(pic_save_path+ f"diff_und_t.png",diff_und, vmin=0, vmax=0.1)
plt.show()


print(wide_image_gnd.shape, wide_image_pred.shape)
diff_gnd = cv2.absdiff(wide_image_gnd, wide_image_gnd)
plt.imshow(diff_gnd, vmin=0, vmax=0.1)
plt.imsave(pic_save_path+ f"diff_gnd_t.png",diff_gnd, vmin=0, vmax=0.1)
plt.show()

print(wide_image_gnd.shape, wide_image_pred.shape)
diff_pred = cv2.absdiff(wide_image_gnd, np.array(wide_image_pred))
print(diff_pred.max(), diff_pred.mean())
plt.imshow(diff_pred, vmin=0, vmax=0.1)
plt.imsave(pic_save_path+ f"diff_pred_{sample_methods}_t.png",diff_pred, vmin=0, vmax=0.1)
plt.show()



# draw spatial pic

Pred_sp = abs(pred[0,slice,...]).astype(np.float32)
plt.imshow(Pred_sp, cmap='gray')
plt.show()

plt.axis('off') 
gnd_sp = abs(gnd[0,slice,...]).astype(np.float32)
plt.axvline(x=int(w/2), color='w', linestyle='--', linewidth=1)
plt.imshow(gnd_sp, cmap='gray')
plt.savefig(pic_save_path+ "image_gnd_line.png", bbox_inches='tight', pad_inches=0, dpi=300)
plt.show()

und_sp = abs(und[0,slice,...]).astype(np.float32)
plt.imshow(und_sp, cmap='gray')
plt.show()

plt.imsave(pic_save_path+ f"img_pred_{sample_methods}.png",Pred_sp, cmap='gray')
plt.imsave(pic_save_path+ f"img_gnd.png",gnd_sp, cmap='gray')
plt.imsave(pic_save_path+ f"img_und.png",und_sp, cmap='gray')

diff_pred = cv2.absdiff(gnd_sp, Pred_sp)
print(diff_pred.max(), diff_pred.mean())
plt.imshow(diff_pred, vmin=0, vmax=0.1)
plt.imsave(pic_save_path+ f"diff_pred_{sample_methods}.png",diff_pred, vmin=0, vmax=0.1)
plt.show()

diff_und = cv2.absdiff(gnd_sp, und_sp)
print(diff_und.max(), diff_und.mean())
plt.imshow(diff_und, vmin=0, vmax=0.1)
plt.imsave(pic_save_path+ f"diff_und.png",diff_und, vmin=0, vmax=0.1)
plt.show()

diff_gnd = cv2.absdiff(gnd_sp, gnd_sp)
plt.imshow(diff_gnd, vmin=0, vmax=0.1)
plt.imsave(pic_save_path+ f"diff_gnd.png",diff_gnd, vmin=0, vmax=0.1)
plt.show()


# diff_gnd = diff_gnd.transpose(1,0)
