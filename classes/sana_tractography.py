
# installed modules
import cv2 as cv
import numpy as np

# debugging modules
from matplotlib import pyplot as plt

class STA:
    def __init__(self, sigma):
        self.sigma = sigma
    #
    # end of constructor

    def run(self, frame, debug=False):

        # f(x,y)
        img = frame.img
        
        # TODO: could do 45 and 135 degree partial derivatives?
        # d/dx(f(x,y)), d/dy(f(x,y))
        imgx = cv.Sobel(img, cv.CV_32F, 1, 0, 3)
        imgy = cv.Sobel(img, cv.CV_32F, 0, 1, 3)    

        # TODO: comment this!
        imgxx = cv.multiply(imgx, imgx)
        imgxy = cv.multiply(imgx, imgy)
        imgyy = cv.multiply(imgy, imgy)    

        J11 = cv.boxFilter(imgxx, cv.CV_32F, self.sigma)
        J12 = cv.boxFilter(imgxy, cv.CV_32F, self.sigma)
        J22 = cv.boxFilter(imgyy, cv.CV_32F, self.sigma)

        # calculate coherence from the J terms
        tmp1 = J11 + J22
        tmp2 = J11 - J22
        tmp2 = cv.multiply(tmp2, tmp2)
        tmp3 = cv.multiply(J12, J12)
        tmp4 = np.sqrt(tmp2 + 4.0 * tmp3)
        lambda1 = 0.5 * (tmp1 + tmp4)
        lambda2 = 0.5 * (tmp1 - tmp4)
        self.coh = cv.divide(lambda1 - lambda2, lambda1 + lambda2)

        # calculate angle from the J terms
        self.ang = 0.5 * cv.phase(J22 - J11, 2.0 * J12, angleInDegrees=True)

        # make sure there are no nan's
        self.coh = np.nan_to_num(self.coh)[:,:,None]
        self.ang = np.nan_to_num(self.ang)[:,:,None]

        if debug:
            fig, axs = plt.subplots(1,2)
            axs[0].imshow(self.coh)
            axs[1].imshow(self.ang)
            plt.show()
    #
    # end of run
#
# end of STA
