
# installed modules
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import sana.threshold
import sana.image

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

def run_directional_analysis(coh, ang, debug=False):

    # restrict angles between 0 and 180 (fold unit circle SW -> NE)
    ang.img = np.where(ang.img < 0, ang.img+180, ang.img)
    ang.img = np.where(ang.img > 180, ang.img-180, ang.img)

    # restrict angles between 0 and 90 (fold unit circle W -> E)
    ang.img = 90 - np.abs(90 - ang.img)
    
    # normalize the angle value
    ang.img = ang.img / 90

    # verticality is closeness to 1.0
    v_ang = ang.img.copy()

    # horizontality is closeness to 0.0
    h_ang = 1 - ang.img

    # diagonality is closeness to 0.5
    d_ang = (0.5 - np.abs(0.5 - ang.img)) / 0.5
                    
    directional_ang = np.concatenate([v_ang, h_ang, d_ang], axis=2)
    prob = coh.img * directional_ang
    #prob = np.nan_to_num(prob)
    #prob = (prob - np.mean(prob, axis=(0,1), keepdims=True)) / np.std(prob, axis=(0,1), keepdims=True)
    prob /= np.max(prob)

    # TODO: triangular method might not be suited here... pretty obvious V fibers are being missed
    # TODO: can we just do accumulation of V/H/D i think not because DAB intensity is involved...
    th = []
    sta_strictness = [-0.5, -0.5, -0.5]
    for i in range(prob.shape[2]):
        p = prob[:,:,i].flatten()
        hist = np.histogram(p, 255)[0].astype(float)
        th.append(sana.threshold.triangular_method(hist, strictness=sta_strictness[i]) / 255)

    V = sana.image.Frame(((prob[:,:,0] >= th[0]) & (np.argmax(prob, axis=2) == 0)).astype(np.uint8))
    H = sana.image.Frame(((prob[:,:,1] >= th[1]) & (np.argmax(prob, axis=2) == 1)).astype(np.uint8))
    D = sana.image.Frame(((prob[:,:,2] >= th[2]) & (np.argmax(prob, axis=2) == 2)).astype(np.uint8))

    if debug:
        fig, axs = plt.subplots(2,3, sharex=True, sharey=True)
        axs = axs.ravel()
        axs[0].imshow(coh.img, cmap='gray')
        axs[1].imshow(ang.img, cmap='gray')
        axs[2].imshow(prob/0.2, cmap='gray')
        axs[3].imshow(V.img, cmap='gray')
        axs[4].imshow(H.img, cmap='gray')
        axs[5].imshow(D.img, cmap='gray')
    
    return V, H, D
