from ultralytics import YOLO
import torch

print(torch.cuda.is_available())

# Load a modelconda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
model = YOLO('yolov8x-pose-p6.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom model

# Predict with the model
results = model(source='Videos/1_Petzold Luca_lq.mp4', show=True, save=True)  # predict on an image