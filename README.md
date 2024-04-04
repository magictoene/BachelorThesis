# Bachelor Thesis & Cross-Country Skiing Analytics (CCSA)

## Introduction
This repository contains the code base for my Bachelor Thesis and the Cross-Country Skiing Analytics (CCSA) project. The aim of the thesis is to develop a labelling tool for cross-country skiing related video footage, focusing on aspects like image matching and pose estimation, utilizing Python for data analysis and processing.

## Installation
To use this project, clone the repository and set up the environment:
```bash
git clone https://github.com/magictoene/BachelorThesis.git
cd BachelorThesis
conda env create -f environment.yml
conda activate [env_name]
```

## Features
Feature extraction from cross-country skiing video data.



## Documentation
The repository includes Python scripts for data processing and analysis, as well as an environment file for setting up the project. For detailed documentation, refer to in-line comments within each script.

## Flowchart

![Beige Colorful Minimal Flowchart Infographic Graph (2)](https://github.com/magictoene/BachelorThesis/assets/101808762/10795bfc-1ac6-4575-8d4e-4f546db2617e)

## Step-by-Step Introduction

### 1.1 Feature Extraction Algorithm (featureExtraction.py)

#### 1.1.1 Frame Extraction

#### 1.1.2 Image Comparison

Good Match:
![comparison_result_3](https://github.com/magictoene/BachelorThesis/assets/101808762/7f242dfa-b280-48c2-991a-0d8f7fb16186)
<img src="https://github.com/magictoene/BachelorThesis/assets/101808762/7f242dfa-b280-48c2-991a-0d8f7fb16186" width="100" height="100">

Bad Match:
![comparison_result](https://github.com/magictoene/BachelorThesis/assets/101808762/ecc257fb-8439-40fa-85b8-a57cd85c04eb)


#### 1.1.3. Label Declaration 


### 1.2 Feature Extraction (poseEstimation.py)

#### 1.2.1 Object Detection

#### 1.2.2 Pose Estimation


### 2 Neural Network Training


## Reasoning behind Library Choices

OpenCV ORB

FuzzyWuzzy

Ultralytics YOLOv8

JSON





