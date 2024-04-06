from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import img_as_float
from skimage import img_as_float, io, transform
from skimage.metrics import peak_signal_noise_ratio
from matplotlib import pyplot as plt
from scipy import ndimage as nd


noisy_img = img_as_float(io.imread("images/MRI_images/mri_noisy.jpg", as_gray=True))
ref_img = img_as_float(io.imread("images/MRI_images/mri_clean.jpg", as_gray=True))

desired_size = (200, 200)

noisy_img = transform.resize(noisy_img, desired_size, mode='reflect', anti_aliasing=True)
ref_img = transform.resize(ref_img, desired_size, mode='reflect', anti_aliasing=True)

######################## Bilateral ##############################
sigma_est = estimate_sigma(noisy_img, average_sigmas=True)

denoise_bilateral = denoise_bilateral(noisy_img, sigma_spatial=15)

plt.imsave("images/MRI_images/bilateral_smoothed.jpg", denoise_bilateral, cmap='gray')

noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
bilateral_cleaned_psnr = peak_signal_noise_ratio(ref_img, denoise_bilateral)

print("Bilateral Denoising:")
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", bilateral_cleaned_psnr)

from skimage.metrics import structural_similarity as ssim

noise_ssim, _ = ssim(ref_img, noisy_img, full=True, win_size=3, data_range=1.0)
cleaned_ssim, _ = ssim(ref_img, denoise_bilateral, full=True, win_size=3, data_range=1.0)

print("SSIM of input noisy image = ", noise_ssim)
print("SSIM of cleaned image = ", cleaned_ssim)
