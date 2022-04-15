# Facial-Detection-Methods

## ENCM 509 Final Project
### Aidan Johnson
### Cameron Faith

![GitHub](https://img.shields.io/github/license/aidanjoh/Facial-Detection-Methods)

![Most recent commit](https://img.shields.io/github/last-commit/aidanjoh/Facial-Detection-Methods)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/aidanjoh/Facial-Detection-Methods)

## Description

The purpose of this project is to explore three different methods of facial detection and evaluate their performance. In this analysis, we will be using eight of our own personal images to compare different facial detection approaches. Each image will contain different elements that will challenge the three techniques and provide insight into the strengths and weaknesses of each face detection method. 

## Table of Contents

- [Installation](#Installation)
- [Results](#Results)

## Installation
### Cloning the Github Repo

First clone the Github Repository to where you would like to store and run the code from. If you choose to only download the juypter notebook instead of the repository make sure that the directory contains:
- A folder called Images which includes all the testing images we used for our project
- The two XML files for the classifiers, "Haarcascade_frontalface_default.xml" and "lbpcascade_frontalface_improved.xml"
- A csv file storing the ground truth bounding boxes called "updatedGroundTruthBBox.csv"
- A yaml file called finalProject.yml that can be used to recreated the anaconda environment that we were using

### Setting up the Conda Environment
To set up the anaconda environment that we used for our project please follow these steps:
1. First download and install Anaconda.
2. Second create an anaconda environment from the provided environment yaml file either through the Anaconda navigator GUI or through the command line running the command: `conda env create --file finalProject.yml`

### Running the Code
To run the code open up the juypter notebook called FacialDetection-AidanJohnson&CameronFaith, select the anaconda environment that was just created and select run all. 

## Results
### Haar Cascade Precision vs Recall Plot

![Haar Cascade Precision vs Recall](docs/HaarCascadePrecisionVsRecall.png?raw=true "Haar Cascade Precision vs Recall")

### LBP Precision vs Recall Plot

![LBP Precision vs Recall Plot](docs/LBPPrecisionVsRecall.png?raw=true "LBP Precision vs Recall Plot")

### HOG + SVM Precision vs Recall Plot

![HOG + SVM Precision vs Recall Plot](docs/HOGPrecisionVsRecall.png?raw=true "HOG + SVM Precision vs Recall Plot")

### Precsion vs Recall Plot of All 3 Methods

![Precsion vs Recall Plot of All 3 Methods](docs/All3.png?raw=true "Precsion vs Recall Plot of All 3 Methods")
