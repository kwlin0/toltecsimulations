# TolTEC Simulations
This repository contains the documentation and archived code to create TolTEC simulated maps using the SIDES catalog (Bethermin et al. 2017) and the source extraction algorithm used for analyzing these maps.

There are three main parts to the program:
* Creating maps (signal, noise, combine)
* Source extraction (based on chosen flux limit)
* Catalog matching (based on chosen search radius)

## Creating Maps

Depends on the following files:
* AzTEC PSF (.fits)
* SCCAT data table

Output products: convolved signal map, convolved noise map, combined signal + noise simulation

The SCCAT data table used to create our maps has been flux cut, meaning that we do not inject sources < 0.01 mJy at the 1.1 mm wavelength passband into the maps. This is because sources under this flux limit would not be detectable above the noise level and by removing these extremely faint sources, we can significantly reduce the file size of the input catalog.

## Source Extraction

Depends on the following files:
* Simulated map (.fits)

Output products: list of pixels above imposed flux limit, extracted source catalog with positions and fluxes

## Catalog Matching

Depends on the following files:
* Extracted source catalog with positions and fluxes
* SCCAT data table
