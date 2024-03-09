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

denoise_TV = denoise_tv_chambolle(noisy_img, weight=0.3)

noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
TV_cleaned_psnr = peak_signal_noise_ratio(ref_img, denoise_TV)

print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", TV_cleaned_psnr)

plt.imsave("images/MRI_images/TV_smoothed.jpg", denoise_TV, cmap='gray')
