#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" tv.py - testing reconstruction using total variation regularization,
using more or less from PyLops docs:
https://pylops.readthedocs.io/en/latest/tutorials/ctscan.html

Steps:
    * model raw data obtained from a CT scan using radon transform
    * invert the sinogram using a TV-regularized solver Split-Bregman

The scikit-image library is used only for shepp_logan_phantom and rescale.
"""

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale
import pylops

def tv_reg_reconstruction(ntheta, mangle, fname):
    ''' Recunstructs Shepp-Logan Phantom using total variation regularized
    reconstruction.
    Args:
        * ntheta = 40            # number of angles from 0 to maximal angle
        * mangle = np.pi         # maximal angle
        * fname = "-full.png"    # to save data and reconstruction to ../figures
    '''
    image = shepp_logan_phantom()
    image = rescale(image, scale=0.4, mode='reflect')

    x = image.T
    x = x / x.max()
    nx, ny = x.shape

    theta = np.linspace(0.0, mangle, ntheta, endpoint=False)

    @jit(nopython=True)
    def radoncurve(x, r, theta):
        return (
            (r - ny // 2) / (np.sin(theta) + 1e-15)
            + np.tan(np.pi / 2.0 - theta) * x
            + ny // 2
        )

    RLop = pylops.signalprocessing.Radon2D(
        np.arange(ny),
        np.arange(nx),
        theta,
        kind=radoncurve,
        centeredh=True,
        interp=False,
        engine="numba",
        dtype="float64",
    )

    y = RLop.H * x
    # xrec = RLop * y # apply the adjoint - this is the first step of FBP

    ## sinogram - graphic representation of the raw data from a CT scan
    fig, ax = plt.subplots(1, figsize=(10, 4))
    fig.tight_layout()
    plt.axis('off')
    ax.imshow(y.T, cmap="gray")
    plt.savefig("../figures/data"+fname, facecolor="white", transparent=True,
      dpi=200, bbox_inches='tight')

    ## try to invert the operator using total variation regularization
    Dop = [
        pylops.FirstDerivative(
            (nx, ny), axis=0, edge=True, kind="backward", dtype=np.float64
        ),
        pylops.FirstDerivative(
            (nx, ny), axis=1, edge=True, kind="backward", dtype=np.float64
        ),
    ]
    D2op = pylops.Laplacian(dims=(nx, ny), edge=True, dtype=np.float64)

    mu = 1.5
    lamda = [1.0, 1.0]
    niter = 3
    niterinner = 4

    xinv = pylops.optimization.sparsity.splitbregman(
        RLop.H,
        y.ravel(),
        Dop,
        niter_outer=niter,
        niter_inner=niterinner,
        mu=mu,
        epsRL1s=lamda,
        tol=1e-4,
        tau=1.0,
        show=False,
        **dict(iter_lim=20, damp=1e-2)
    )[0]
    xinv = np.real(xinv.reshape(nx, ny))

    fig, ax = plt.subplots(1, figsize=(10, 4))
    fig.tight_layout()
    plt.axis('off')
    ax.imshow(xinv.T, vmin=0, vmax=1, cmap="gray")
    plt.savefig("../figures/TV"+fname, facecolor="white", transparent=True,
        dpi=200, bbox_inches='tight')


if __name__ == "__main__":
    np.random.seed(2727)
    print(pylops.__version__) # 1.18.1.dev158+g70b75f1
    tv_reg_reconstruction(40, np.pi, "-full.png")
    tv_reg_reconstruction(20, np.pi/2, "-half.png")
