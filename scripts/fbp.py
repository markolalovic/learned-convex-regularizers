#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" fbp.py - filtered back projection, more or less from scikit-image examples:
https://scikit-image.org/docs/stable/auto_examples/transform/plot_radon_transform.html
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale
from skimage.transform import iradon

def savegroundtruth():
    image = shepp_logan_phantom()
    image = rescale(image, scale=0.4, mode='reflect')

    fig, ax = plt.subplots(1, figsize=(10, 4))
    fig.tight_layout()
    plt.axis('off')

    ax.imshow(image, vmin=0, vmax=1, cmap="gray")
    plt.savefig('../figures/GT.png', facecolor="white", transparent=True,
        dpi=200, bbox_inches='tight')

def reconstruct(ntheta, mangle, fname):
    ''' Recunstructs Shepp-Logan Phantom
    Args:
        * ntheta = 40 # number of angles from 0 to maximal angle
        * mangle = 180 # maximal angle
        * fname = "FBP-full.png" # to save in ../figures
    '''
    image = shepp_logan_phantom()
    image = rescale(image, scale=0.4, mode='reflect')

    theta = np.linspace(0., mangle, ntheta, endpoint=False)
    sinogram = radon(image, theta=theta) # forward projection


    ## reconstruction using filtered back projection - adjoint with ramp filter
    reconstruction_fbp = iradon(sinogram, theta=theta, filter_name='ramp')
    error = reconstruction_fbp - image
    print(f'FBP rms reconstruction error: {np.sqrt(np.mean(error**2)):.3g}')

    ## plot the results
    imkwargs = dict(vmin=-0.2, vmax=0.2)
    fig, ax = plt.subplots(1, figsize=(10, 4), sharex=True, sharey=True)
    fig.tight_layout()
    plt.axis('off')
    ax.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
    plt.savefig("../figures/"+fname, facecolor="white", transparent=True, dpi=200, bbox_inches='tight')

if __name__ == "__main__":
    savegroundtruth()
    reconstruct(40, 180, "FBP-full.png") # RMS = 0.0514
    reconstruct(20, 90,  "FBP-half.png") # RMS = 0.182
