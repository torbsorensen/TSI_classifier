# TSI_classifier (in the making...)
A short guide on how to build a convolutional neural network to classify images from the Total Sky Imager on Summit Station, Greenland, in fogbow/icebow/nobow

## Introduction
This short guide is a result from a summer internship completed in 2021 at [SENSE - Centre for Satellite Data in Environmental Science](https://eo-cdt.org) at the University of Leeds under supervision from Dr Ryan R. Neely and Heather Guy. The overall goal of the project was to investigate Greenlandic fog, and we decided to use the large amount of photos from the Total Sky Imager to make a timeseries of when fog occured and whether it formed by liquid or frozen droplets. 

To solve this problem I decided to try applying a convolutional neural network, and as a undergraduate student with some coding but no machine learning experience, the learning curve has been steep. The methods presented in this project is very likely **not** the best practice and someone with a ML/data science background might get a headache from reading it, but it is the product of my 6 weeks of learning deep learning and it solves the problem in a reasonable manner. 

Should you have any suggestions or questions, please do get in contact on tor.sorensen.18@ucl.ac.uk or elsewhere. Let's get going!

## System requirements

The work has been done in Python (3.7.10) on a Jupyter Notebook using Tensorflow (2.5.0), Keras and a bit of scikit-learn (0.24.2). Other used packages are numpy (1.21.1), matplotlib (3.3.4), os, PIL (8.3.1), pathlib and shutil. A full list of packages installed in the environment is in the file environment.txt.

## Data

The images analysed are taken by the [Total Sky Imager (TSI)](ftp://ftp1.esrl.noaa.gov/psd3/arctic/summit/tsi/0_docs/Summit_Datagrams_totalskyimager.pdf) on the Summit Station, Greenland, an instrument run from the roof of the ICECAPS Observatory. Below are three examples of photos taken by the TSI, one with a fogbow, one with frozen fogbow and one without any bow:

![fogbow](/data/examples/fogbow.jpg) ![iceoptics](/data/examples/iceoptics.jpg) ![no_optics](/data/examples/no_optics.jpg) 

A small training dataset of 218 labeled photos is avaliable under /data/trainingset. The full training set of 18129 labeled photos can be obtained by contacting me.

## Code

## Results
