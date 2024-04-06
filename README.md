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
The figure below shows a flowchart of the basic code concept.

![Beige Colorful Minimal Flowchart Infographic Graph (2)](https://github.com/magictoene/BachelorThesis/assets/101808762/10795bfc-1ac6-4575-8d4e-4f546db2617e)


## Step-by-Step Introduction

### 1.1 Feature Extraction Algorithm (featureExtraction.py)

Currently there only exist videos and corresponding label images for each dataset. Compared to the video, the label images only include the narrow view of the cross-country skier. For neural network training, a more comprehensive view of the label images is necessary. Thefore, the small sized labels need to be matched to the video frames to get label images that span across the whole screen.

The feature extraction algorithm consists of two major steps: matching label images to extracted video frames and then running pose estimation on the latter. Result of the feature extraction is a dataframe that contains key joint data relevant for the neural network training.

#### 1.1.1 Frame Extraction
For each video, an own folder is created that stores the frames extracted with the help of OpenCV.

#### 1.1.2 Image Comparison
OpenCV incorporates the ORB (Oriented FAST and Rotated BRIEF) algorithm, that locates key points in images. The key points are then used to match each of the extracted frames to each of the corresponding labelled images. The matches are then filtered and underdo a homography check. If the paired key points of the two images have a similar spatial orientation when rotated, the pairing is considered a match. The frame then becomes the new label, until  a better match is found.

Below is an example that shows how ORB matches the image key points and creates pairings. The first image shows a good match, where frame '0031.jpg' becomes label '3'. The second image shows an attempt of matching two not-so-similar images. 

<img src="https://github.com/magictoene/BachelorThesis/assets/101808762/7f242dfa-b280-48c2-991a-0d8f7fb16186" width="480" height="360">
<img src="https://github.com/magictoene/BachelorThesis/assets/101808762/ecc257fb-8439-40fa-85b8-a57cd85c04eb" width="480" height="360">


After finishing a folder, the label information is stored in a JSON file:
```bash
{
    "1": "Frames\\100_Thannheimer Germana_lq\\0023.jpg",
    "2": "Frames\\100_Thannheimer Germana_lq\\0025.jpg",
    "3": "Frames\\100_Thannheimer Germana_lq\\0031.jpg",
    "4": "Frames\\100_Thannheimer Germana_lq\\0040.jpg",
    "5": "Frames\\100_Thannheimer Germana_lq\\0047.jpg",
    "0": [
        "Frames\\100_Thannheimer Germana_lq\\0000.jpg",
        .
        "Frames\\100_Thannheimer Germana_lq\\xxxx.jpg
         ]
}
```


### 1.2 Feature Extraction (poseEstimation.py)

#### 1.2.1 Object Detection

#### 1.2.2 Pose Estimation


### 2 Neural Network Training


## Reasoning behind Library Choices

OpenCV ORB

FuzzyWuzzy

Ultralytics YOLOv8





