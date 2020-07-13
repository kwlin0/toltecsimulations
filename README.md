# TolTEC Simulations
This repository contains the documentation and archived code to create TolTEC simulated maps using the SIDES catalog (Bethermin et al. 2017) and the source extraction algorithm used for analyzing these maps.

There are three main parts to the program:
* Creating maps (signal, noise, combine)
* Source extraction (based on chosen flux limit)
* Catalog matching (based on chosen search radius)

## System Requirements
The codes have been tested and verified for MacOS Catalina running Python 3.6.8. Packages used were numpy v1.16.2, scipy v1.2.1, matplotlib v3.0.3, astropy v3.1.2, FITS_tools v0.2.

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

Output products: list of input and output fluxes for the single matches and multiple matches, list of output fluxes for the no match sources, summary of completeness results for the run

The output products from this step are what is fed into the data analysis with the completeness and flux boosting plots.
