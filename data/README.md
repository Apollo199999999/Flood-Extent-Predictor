# Data creation and collation
* This folder contains all the code necessary to create the dataset needed for training and testing.
* Note that the scripts here rely on Google Earth Engine to access data, hence you will need a Google Account with Google Earth Engine available and with sufficient space available to export the large feature maps to.
* The raw TIFF files, as well as the processed tiles for model training, is available on HuggingFace (soon).
    * Raw TIFF files: https://huggingface.co/datasets/MatthiasWang/geotiffs
    * Processed tiles: https://huggingface.co/datasets/MatthiasWang/processed-tiles

## Flood events
For training and testing, 8 prominent flood events within the last 2 decades or so were selected from the Global Flood Database, with 6 set aside for training and 2 set aside for testing for model evaluation. Ground truth labels obtained are in 250m resolution.

### Training
* 2005 Yangtze River Floods
* 2007 South Asian/Bangladesh Floods (Most population exposed)
* 2007 Tabasco Floods
* 2009 Red River Floods
* 2015 Paraguay and Uruguay El Nino Floods
* 2017 California Floods

### Testing
* 2011 Thailand Floods
* 2018 Nigeria Floods

## Flood conditioning factors
This dataset uses Google Earth Engine to obtain feature maps for the following flood conditioning factors that have been identified to be more pertinent for flood detection. These feature maps are stored in 25m resolution. Where possible, these feature maps are obtained from dates within 6 months prior to the flood event.
* Altitude (DEM)
* Slope (multiplied by 1000 and stored as integer)
* Land use
* Normalized Difference Vegetation Index (NDVI) (multiplied by 1000 and stored as integer)
* Distance from permanent water (multiplied by 1000 and stored as integer)


