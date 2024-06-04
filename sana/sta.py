
# installed modules
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def run_directional_sta(frame, sigma):
    # TODO: frame type checking

    coh, ang = run_sta(frame, sigma)

    v_ang = (90 - np.abs(90 - ang)) / 90
    h_ang = np.abs(90-ang) / 90
    d_ang = (45 - np.abs(45 - np.abs(90 - ang))) / 45

    directional_ang = np.concatenate([v_ang, h_ang, d_ang], axis=2)

    return coh, directional_ang

# https://docs.opencv.org/4.x/d4/d70/tutorial_anisotropic_image_segmentation_by_a_gst.html
def run_sta(frame, sigma):
    # TODO: frame type checking
    
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

    J11 = cv.boxFilter(imgxx, cv.CV_32F, sigma)
    J12 = cv.boxFilter(imgxy, cv.CV_32F, sigma)
    J22 = cv.boxFilter(imgyy, cv.CV_32F, sigma)

    # calculate coherence from the J terms
    tmp1 = J11 + J22
    tmp2 = J11 - J22
    tmp2 = cv.multiply(tmp2, tmp2)
    tmp3 = cv.multiply(J12, J12)
    tmp4 = np.sqrt(tmp2 + 4.0 * tmp3)
    lambda1 = 0.5 * (tmp1 + tmp4)
    lambda2 = 0.5 * (tmp1 - tmp4)
    coh = cv.divide(lambda1 - lambda2, lambda1 + lambda2)

    # calculate angle from the J terms
    ang = 0.5 * cv.phase(J22 - J11, 2.0 * J12, angleInDegrees=True)

    # make sure there are no nan's
    coh = np.nan_to_num(coh)[:,:,None]
    ang = np.nan_to_num(ang)[:,:,None]

    return coh, ang
