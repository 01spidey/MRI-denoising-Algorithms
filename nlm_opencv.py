#NLM opencv
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html
# cv2.fastNlMeansDenoising() - works with a single grayscale images
# cv2.fastNlMeansDenoisingColored() - works with a color image.

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_ubyte, img_as_float
from matplotlib import pyplot as plt
from skimage import io, transform
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

noisy_img = img_as_float(io.imread("images/MRI_images/mri_noisy.jpg", as_gray=True))
ref_img = img_as_float(io.imread("images/MRI_images/mri_clean.jpg", as_gray=True))

desired_size = (200, 200)

noisy_img = transform.resize(noisy_img, desired_size, mode='reflect', anti_aliasing=True)
ref_img = transform.resize(ref_img, desired_size, mode='reflect', anti_aliasing=True)

# Convert the image to uint8
noisy_img_uint8 = img_as_ubyte(noisy_img)

# Apply NLM denoising
NLM_CV2_denoise_img = cv2.fastNlMeansDenoising(noisy_img_uint8, None, 3, 7, 21)

# Convert back to float for saving and visualization
NLM_CV2_denoise_img_float = img_as_float(NLM_CV2_denoise_img)

plt.imsave("images/MRI_images/NLM_CV2_denoised.jpg", NLM_CV2_denoise_img_float, cmap='gray')

noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
NLM_CV2_cleaned_psnr = peak_signal_noise_ratio(ref_img, NLM_CV2_denoise_img_float)

print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", NLM_CV2_cleaned_psnr)

from skimage.metrics import structural_similarity as ssim

noise_ssim, _ = ssim(ref_img, noisy_img, full=True, win_size=3, data_range=1.0)
cleaned_ssim, _ = ssim(ref_img, NLM_CV2_denoise_img_float, full=True, win_size=3, data_range=1.0)

print("SSIM of input noisy image = ", noise_ssim)
print("SSIM of cleaned image = ", cleaned_ssim)