import os
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy import misc

spectrogram = misc.imread("data_image/f1_mic_5_4kHz.png")

plt.imshow(spectrogram)
plt.show()