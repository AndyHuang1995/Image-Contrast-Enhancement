import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure as ex
import imageio
import sys

def he(img):
    if(len(img.shape)==2):      #gray
        outImg = ex.equalize_hist(img[:,:])*255 
    elif(len(img.shape)==3):    #RGB
        outImg = np.zeros((img.shape[0],img.shape[1],3))
        for channel in range(img.shape[2]):
            outImg[:, :, channel] = ex.equalize_hist(img[:, :, channel])*255

    outImg[outImg>255] = 255
    outImg[outImg<0] = 0
    return outImg.astype(np.uint8)

def main():
    img_name = sys.argv[1]
    img = imageio.imread(img_name)
    result = he(img)
    plt.imshow(result)
    plt.show()

if __name__ == '__main__':
    main()