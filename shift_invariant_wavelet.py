
import matplotlib.pyplot as plt

from skimage.restoration import denoise_wavelet, cycle_spin
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
from skimage import io, transform

noisy_img = img_as_float(io.imread("images/MRI_images/mri_noisy.jpg", as_gray=True))
ref_img = img_as_float(io.imread("images/MRI_images/mri_clean.jpg", as_gray=True))

desired_size = (200, 200)

noisy_img = transform.resize(noisy_img, desired_size, mode='reflect', anti_aliasing=True)
ref_img = transform.resize(ref_img, desired_size, mode='reflect', anti_aliasing=True)

denoise_kwargs = dict(wavelet='db1', 
                      method='BayesShrink',
                      rescale_sigma=True)

all_psnr = []
max_shifts = 3     #0, 1, 3, 5

Shft_inv_wavelet = cycle_spin(noisy_img, 
                              num_workers=1,
                              func=denoise_wavelet, 
                              max_shifts = max_shifts, 
                              func_kw=denoise_kwargs)

noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
shft_cleaned_psnr = peak_signal_noise_ratio(ref_img, Shft_inv_wavelet)

print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", shft_cleaned_psnr)

plt.imsave("images/MRI_images/Shift_Inv_wavelet_smoothed.jpg", Shft_inv_wavelet, cmap='gray')

from skimage.metrics import structural_similarity as ssim

noise_ssim, _ = ssim(ref_img, noisy_img, full=True, win_size=3, data_range=1.0)
cleaned_ssim, _ = ssim(ref_img, Shft_inv_wavelet, full=True, win_size=3, data_range=1.0)

print("SSIM of input noisy image = ", noise_ssim)
print("SSIM of cleaned image = ", cleaned_ssim)