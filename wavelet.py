from skimage import img_as_float, io, transform
from skimage.metrics import peak_signal_noise_ratio
from matplotlib import pyplot as plt
from scipy import ndimage as nd
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)

noisy_img = img_as_float(io.imread("images/MRI_images/mri_noisy.jpg", as_gray=True))
ref_img = img_as_float(io.imread("images/MRI_images/mri_clean.jpg", as_gray=True))

desired_size = (200, 200)

noisy_img = transform.resize(noisy_img, desired_size, mode='reflect', anti_aliasing=True)
ref_img = transform.resize(ref_img, desired_size, mode='reflect', anti_aliasing=True)

wavelet_smoothed = denoise_wavelet(noisy_img,
                                   method='BayesShrink', 
                                   mode='soft',
                                   rescale_sigma=True)

noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
Wavelet_cleaned_psnr = peak_signal_noise_ratio(ref_img, wavelet_smoothed)

print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", Wavelet_cleaned_psnr)

plt.imsave("images/MRI_images/wavelet_smoothed.jpg", wavelet_smoothed, cmap='gray')