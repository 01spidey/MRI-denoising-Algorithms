#Anisotropic Diffusion

import matplotlib.pyplot as plt
import cv2
from skimage import io, transform
from medpy.filter.smoothing import anisotropic_diffusion
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio


noisy_img = img_as_float(io.imread("images/MRI_images/mri_noisy.jpg", as_gray=True))
ref_img = img_as_float(io.imread("images/MRI_images/mri_clean.jpg", as_gray=True))

desired_size = (200, 200)

noisy_img = transform.resize(noisy_img, desired_size, mode='reflect', anti_aliasing=True)
ref_img = transform.resize(ref_img, desired_size, mode='reflect', anti_aliasing=True)

# niter= number of iterations
#kappa = Conduction coefficient (20 to 100)
#gamma = speed of diffusion (<=0.25)
#Option: Perona Malik equation 1 or 2. A value of 3 is for Turkey's biweight function 
img_aniso_filtered = anisotropic_diffusion(noisy_img, niter=50, kappa=50, gamma=0.2, option=2) 

noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
anisotropic_cleaned_psnr = peak_signal_noise_ratio(ref_img, img_aniso_filtered)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", anisotropic_cleaned_psnr)


plt.imshow(img_aniso_filtered, cmap='gray')
plt.imsave("images/MRI_images/anisotropic_denoised.tif", img_aniso_filtered, cmap='gray')