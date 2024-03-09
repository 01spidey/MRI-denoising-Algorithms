import matplotlib.pyplot as plt
from skimage import io, img_as_float, transform
from skimage.metrics import peak_signal_noise_ratio
import bm3d
import numpy as np

noisy_img = img_as_float(io.imread("images/MRI_images/mri_noisy.jpg", as_gray=True))
ref_img = img_as_float(io.imread("images/MRI_images/mri_clean.jpg", as_gray=True))

desired_size = (200, 200)

noisy_img = transform.resize(noisy_img, desired_size, mode='reflect', anti_aliasing=True)
ref_img = transform.resize(ref_img, desired_size, mode='reflect', anti_aliasing=True)


# BM3D_denoised_image = bm3d.bm3d(noisy_img, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.ALL_STAGES)
BM3D_denoised_image = bm3d.bm3d(noisy_img, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)

noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
BM3D_cleaned_psnr = peak_signal_noise_ratio(ref_img, BM3D_denoised_image)

print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", BM3D_cleaned_psnr)

plt.imsave("images/MRI_images/BM3D_denoised.jpg", BM3D_denoised_image, cmap='gray')