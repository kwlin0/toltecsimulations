#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 18:22:43 2020

@author: klin
"""

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
from scipy.stats import norm


def makeField(area, res):
    # area = square degrees of desired field
    # res = resolution of each pixel in arcseconds
    fieldDim = np.sqrt(area) # dimensions for a square field
    degrees = (res/60)/60 # convert arcseconds to degrees
    pixelNumber = fieldDim/degrees # number of pixels dimension
    fieldArray = np.zeros((int(pixelNumber),int(pixelNumber)))
    return fieldArray


def makeSignalMap(path_to_inputFlux, path_to_PSF, band, FWHM, fieldSize, shift_xy):
    ###############################################################################
    #aztecPSF_orig = fits.open(path_to_PSF)[0]
    aztecPSF_resized = FITS_tools.hcongrid.zoom_fits(path_to_PSF, FWHM/8.5) # scale factor FWHM/8.5
    
    # Reads input fluxes text file
    
    dataObject_cut = ascii.read(path_to_inputFlux, guess=False, format='basic')
    

    newArray = makeField(fieldSize,1.0)
    #newBands = [np.array(dataObject_cut['SCCAT1100']),np.array(dataObject_cut['SCCAT1300']),np.array(dataObject_cut['SCCAT2000'])] old code not used
    
    
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
    #adjCoords = [xInt,yInt,newBands[0],newBands[1],newBands[2]] more old code not used
    
    
    for i in range(len(xInt)):
        newArray[yInt[i]][xInt[i]] =  newArray[yInt[i]][xInt[i]] + np.array(dataObject_cut['SCCAT'+str(int(band*1000))])[i]*1000 # converts to mJy
    
    signalArray = newArray
    
    
    shiftedImage = shift(aztecPSF_resized.data, shift_xy) # 1.1 mm [0.5, 0.5], 1.3 mm [0., 0.], 2.0 mm [0., 0.5]
    
    
    dg_aztec=fftconvolve(signalArray,shiftedImage[2:,0:],mode='same') # [2:,0:]
    dg_aztec_scaled = dg_aztec*(np.max(signalArray)/(np.max(dg_aztec)))
    
    toltecPSF = fits.PrimaryHDU(shiftedImage[2:,0:])
    toltecPSF.writeto('toltec_'+str(int(band*1000))+'_PSF.fits')
    
    signalImage = fits.PrimaryHDU(dg_aztec_scaled)
    signalImage.writeto('toltec_'+str(int(band*1000))+'nonoise_'+str(int(fieldSize))+'sd_s.fits')
    np.savetxt('toltec_'+str(int(band*1000))+'nonoise_'+str(int(fieldSize))+'sd_s.txt', dg_aztec_scaled.flatten()) # saves fluxes as txt file
    
    return signalArray, shiftedImage, dg_aztec_scaled


def makeNoiseMap(path_to_inputFlux, path_to_PSF, band, FWHM, fieldSize, sigma, factor):

    aztecPSF_resized = FITS_tools.hcongrid.zoom_fits(path_to_PSF, FWHM/8.5) 
    
    
    SIZE = fieldSize # arcseconds^2
    PXRES = 1.0
    
    blankMap = makeField(SIZE,PXRES)
    
    
    #newBands = [np.array(dataObject_cut['SCCAT1100']),np.array(dataObject_cut['SCCAT1300']),np.array(dataObject_cut['SCCAT2000'])]
    
    
    # Create WCS object (leave just in case need to output ds9)
    w = wcs.WCS(naxis=2)
    w.wcs.ctype = ["RA---ARC", "DEC--ARC"] # zenithal/azimuthal equidistant
    
    
    
    ###########################################
    
    # 1.1 mm: 5 arcsec fudge factor = 1.19
    # 1.4 mm: 6.3 arcsec fudge factor = 1.5
    # 2.0 mm: 9.5 arcsec fudge factor = 2.25
    
    detectorFWHM = FWHM # arcseconds
    
    beamArea = (np.pi*(detectorFWHM)**2)/(4*np.log(2))
    
    PX_PER_BEAM = beamArea*PXRES
    
    noise = (np.random.normal(0,(sigma/np.sqrt(PX_PER_BEAM))*factor,len(blankMap.flatten())))
    noiseReshaped = np.reshape(noise, (-1, len(blankMap)))
    blankField_noise = blankMap + noiseReshaped
    
    dg_blankField = fftconvolve(blankField_noise,aztecPSF_resized.data,mode='same')
    
    
    
    # Save files as FITS and text
    hdu = fits.PrimaryHDU(dg_blankField)
    hdu.writeto('toltec_'+str(int(band*1000))+'_rms'+str(sigma).replace('.', 'p')+'_'+str(int(fieldSize))+'sd_n.fits')
    np.savetxt('toltec_'+str(int(band*1000))+'_rms'+str(sigma).replace('.', 'p')+'_'+str(int(fieldSize))+'sd_n.txt', dg_blankField.flatten()) # saves fluxes as txt file
    
    return blankField_noise, aztecPSF_resized, dg_blankField

def makeSignalNoiseMap(SignalMap, NoiseMap, path_to_destination):

    finalMap = SignalMap + NoiseMap
    finalMap_FITS = fits.PrimaryHDU(finalMap)
    finalMap_FITS.writeto(path_to_destination+'.fits')
    np.savetxt(path_to_destination+'.txt', finalMap.flatten())
    
    return finalMap

#####################################################################

# PROGRAM INPUTS

"""
path_to_inputFlux = '/Users/klin/Documents/Astro/Spring 2019/newSCCATfluxes.txt'# path to input flux file [str]
path_to_PSF = 'AzTEC_psf.fits'                                                  # path to desired PSF FITS file [str]
band = 1.1                                                                      # mm [float]
FWHM = 5                                                                        # arcsec [int/float] (9.5 arcsec for 2.0 mm, 6.3 arcsec for 1.4 mm, 5 arcsec for 1.1 mm)
fieldSize = 2.0                                                                 # square degrees [float]
shift_xy = [0.5, 0.5]                                                           # px in (x,y) [array], 1.1 mm [0.5, 0.5], 1.3 mm [0.5, 0.], 2.0 mm [0., 0.5]
"""

band = 1.1
FWHM = 5
sigma = 0.025
fieldSize = 2.0
shift_xy = [0.5, 0.5]
factor = 1.19

sigarray, shiftedpsf, dg = makeSignalMap('newSCCATfluxes.txt', 'AzTEC_psf.fits', band, FWHM, fieldSize, shift_xy)
rawNoise, psf, noiseMap = makeNoiseMap('newSCCATfluxes.txt', 'AzTEC_psf.fits', band, FWHM, fieldSize, sigma, factor)
final = makeSignalNoiseMap(dg, noiseMap, 'toltec_'+str(int(band*1000))+'_rms'+str(sigma).replace('.', 'p')+'_'+str(int(fieldSize))+'sd') # do not include file extension in pathname


#####################################################################
# REPLOT FIELD (NOT USING NOISE NOW Feb 2020 update)

plt.figure(1)
fig, ax=plt.subplots(1,3,figsize=(12,4))

sky = ax[0].imshow(sigarray,cmap='viridis',origin='lower')
ax[0].text(70,200,'Sky',color='r',fontsize=15)

filt = ax[1].imshow(shiftedpsf,cmap='viridis',origin='lower')
ax[1].text(10,10,'PSF',color='r',fontsize=15)


result = ax[2].imshow(dg,cmap='viridis',origin='lower')
ax[2].text(60,190,'sp.fftconvolve',color='r',fontsize=15)
plt.tight_layout()
plt.show()


plt.figure(2)
fig, ax=plt.subplots(1,3,figsize=(12,4))

sky = ax[0].imshow(rawNoise,cmap='viridis',origin='lower')
ax[0].text(60,20,'Blank Noise Sky',color='r',fontsize=15)

filt = ax[1].imshow(psf.data,cmap='viridis',origin='lower')
ax[1].text(90,15,'AzTEC PSF (scaled)',color='r',fontsize=15)

result = ax[2].imshow(noiseMap,cmap='viridis', origin='lower')
ax[2].text(25,15,'sp.fftconvolve with noise',color='r',fontsize=15)
plt.tight_layout()
    
plt.figure(3)
data = noiseMap.flatten()

# Fit a normal distribution to the data:
mu, std = norm.fit(noiseMap.flatten())

# Plot the histogram.
plt.hist(noiseMap.flatten(), bins=900, density=True, alpha=0.6, color='g', histtype='step', label='Data')


# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'r--', linewidth=0.5, label='Fit')
title = "Fit results: mu = %.2f,  $\sigma_{calc} = $ %.3f" % (mu, std)
plt.title(title)
plt.xlabel("Flux (Jy)"); plt.ylabel("N (normalized)")
plt.tick_params(direction="in")
plt.legend()
plt.tight_layout()
#plt.savefig("NoiseMapDist_"+"FWHM"+str(detectorFWHM)+"_"+str(sigma)+".pdf")
plt.show()