# CS6010 Final Project: Predicting the Presence of MGMT in Glioblastomas From MRI Scans Using Deep Convolutional Neural Networks

Group Members: Michael Blackwell, Sarath Madapana, Naveem Morla

Based on the Kaggle competition: https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification

## Overview
This project investigates whether there are sufficient feature differences in MGMT positive and MGMT negative glioblastomas to classify them based on MRI scans alone. 

## Data
The dataset is from the Kaggle competition mentioned above. It contains MRI images using four different contrasting methods (FLAIR, T1w, T1wCE, T2w) for each of the 585 patients. For each patient, each scan type consists of a series of anywhere between 20-400 images that can be stacked sequentially to give a rudimentary 3D model of the patient's brain. 

## Method
Our theory was that features could be extracted from these stacked images using convolutional neural networks (CNNs). Since there are four different contrasting types for each patient, our model is essentially four independent CNNs being concatenated and piped into a fifth CNN for the final prediction. Refer to the tensorboard files & runscript for a visual graph of the model structure.

## Results
The model failed to classify MGMT+ vs MGMT- glioblastomas with any accuracy greater than mere chance. While other network structures may have a greater success rate and should be assessed, several submissions from the competition used advanced pre-trained models (like imagenet) with similar results. Since this is a classification task that was previously done using a biopsy, it is possible that the appearance of the two tumors on MRI scans are just not sufficiently different to distinguish between them using these imaging methods.
