#Gaussian
from skimage import img_as_float, io, transform
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
from matplotlib import pyplot as plt
from scipy import ndimage as nd

noisy_img = img_as_float(io.imread("images/MRI_images/MRI_noisy.jpg"))
ref_img = img_as_float(io.imread("images/MRI_images/MRI_clean.jpg"))

desired_size = (200, 200)

noisy_img = transform.resize(noisy_img, desired_size, mode='reflect', anti_aliasing=True)
ref_img = transform.resize(ref_img, desired_size, mode='reflect', anti_aliasing=True)


gaussian_img = nd.gaussian_filter(noisy_img, sigma=5)
plt.imshow(gaussian_img, cmap='gray')
plt.imsave("images/MRI_images/Gaussian_smoothed.jpg", gaussian_img, cmap='gray')

noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
gaussian_cleaned_psnr = peak_signal_noise_ratio(ref_img, gaussian_img)

# print the PSNR
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", gaussian_cleaned_psnr)

from skimage.metrics import structural_similarity as ssim

noise_ssim, _ = ssim(ref_img, noisy_img, full=True, win_size=3, data_range=1.0)
gaussian_ssim, _ = ssim(ref_img, gaussian_img, full=True, win_size=3, data_range=1.0)

print("SSIM of input noisy image = ", noise_ssim)
print("SSIM of cleaned image = ", gaussian_ssim)