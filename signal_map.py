#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised 5/29/20

@author: klin
"""

import numpy as np
from astropy import wcs
import matplotlib.pyplot as plt
from astropy.io import ascii
# from photutils.datasets import (make_random_gaussians_table, make_noise_image, make_gaussian_sources_image)
from scipy.signal import fftconvolve
from scipy.ndimage import shift
from astropy.io import fits
import FITS_tools

# INPUTS

path_to_inputFlux = '/Users/klin/Documents/Astro/Spring 2019/newSCCATfluxes.txt'# path to input flux file [str]
path_to_PSF = 'AzTEC_psf.fits'                                                  # path to desired PSF FITS file [str]
band = 1.1                                                                      # mm [float]
FWHM = 5                                                                        # arcsec [int/float] (9.5 arcsec for 2.0 mm, 6.3 arcsec for 1.4 mm, 5 arcsec for 1.1 mm)
fieldSize = 2.0                                                                 # square degrees [float]
shift_xy = [0.5, 0.5]                                                           # px in (x,y) [array], 1.1 mm [0.5, 0.5], 1.3 mm [0.5, 0.], 2.0 mm [0., 0.5]

###############################################################################
aztecPSF_orig = fits.open(path_to_PSF)[0]
aztecPSF_resized = FITS_tools.hcongrid.zoom_fits(path_to_PSF, FWHM/8.5) # scale factor FWHM/8.5

# Reads input fluxes text file

dataObject_cut = ascii.read(path_to_inputFlux, guess=False, format='basic')

def makeField(area, res):
    # area = square degrees of desired field
    # res = resolution of each pixel in arcseconds
    fieldDim = np.sqrt(area) # dimensions for a square field
    degrees = (res/60)/60 # convert arcseconds to degrees
    pixelNumber = fieldDim/degrees # number of pixels dimension
    fieldArray = np.zeros((int(pixelNumber),int(pixelNumber)))
    return fieldArray

newArray = makeField(fieldSize,1.0)
newBands = [np.array(dataObject_cut['SCCAT1100']),np.array(dataObject_cut['SCCAT1300']),np.array(dataObject_cut['SCCAT2000'])]


# Create WCS object
w = wcs.WCS(naxis=2)
w.wcs.ctype = ["RA---ARC", "DEC--ARC"] # zenithal/azimuthal equidistant

# Gridding from WCS to Cartesian

pixelCoords = w.all_world2pix(np.array(dataObject_cut['RA']), np.array(dataObject_cut['DEC']), 1)
xCoords = pixelCoords[0]
yCoords = pixelCoords[1]

# Broadcast to array the size of desired field
oldXRange = max(xCoords) - min(xCoords)
oldYRange = max(yCoords) - min(yCoords)
xCoordsNew = ((xCoords - min(xCoords))*(len(newArray)-1))/oldXRange
yCoordsNew = ((yCoords - min(yCoords))*(len(newArray)-1))/oldYRange


# Rounds coordinates to nearest integer and converts dtype to integer
xInt = np.rint(xCoordsNew).astype(int)
yInt = np.rint(yCoordsNew).astype(int)

# Array of x, y, and flux densities from passbands
adjCoords = [xInt,yInt,newBands[0],newBands[1],newBands[2]]


for i in range(len(xInt)):
    newArray[yInt[i]][xInt[i]] =  newArray[yInt[i]][xInt[i]] + np.array(dataObject_cut['SCCAT'+str(int(band*1000))])[i]*1000 # converts to mJy

signalArray = newArray


shiftedImage = shift(aztecPSF_resized.data, shift_xy)


dg_aztec=fftconvolve(signalArray,shiftedImage[2:,0:],mode='same') # [2:,0:]
dg_aztec_scaled = dg_aztec*(np.max(signalArray)/(np.max(dg_aztec)))

toltecPSF = fits.PrimaryHDU(shiftedImage[2:,0:])
toltecPSF.writeto('toltec_'+str(int(band*1000))+'_PSF.fits')


signalImage = fits.PrimaryHDU(dg_aztec_scaled)
signalImage.writeto('toltec_'+str(int(band*1000))+'_nonoise_'+str(int(fieldSize))+'sd_s.fits')
np.savetxt('toltec_'+str(int(band*1000))+'_nonoise_'+str(int(fieldSize))+'sd_s.txt', dg_aztec_scaled.flatten()) # saves fluxes as txt file

#####################################################################

plt.figure(1)
fig, ax=plt.subplots(1,3,figsize=(12,4))

sky = ax[0].imshow(signalArray,cmap='viridis',origin='lower')
ax[0].text(70,200,'Sky',color='r',fontsize=15)

filt = ax[1].imshow(shiftedImage,cmap='viridis',origin='lower')
ax[1].text(10,10,'PSF',color='r',fontsize=15)


result = ax[2].imshow(dg_aztec_scaled,cmap='viridis',origin='lower')
ax[2].text(60,190,'sp.fftconvolve',color='r',fontsize=15)
#divider = make_axes_locatable(ax[2])
#cax = divider.append_axes("right", size="3%", pad=0.1)
#plt.colorbar(result, cax=cax)
plt.tight_layout()
plt.show()