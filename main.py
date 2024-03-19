import os
import cv2
from test import extract_frames, compare_images, get_filepaths


video_path = "Video/1_Petzold Luca_lq.mp4"
frames_dir = "Frames"

# count = extract_frames(video_path, frames_dir, overwrite=True, start=-1, end=-1, every=1)

target_image = "Pictures/1_Petzold Luca_004.jpg"

# Example usage:

image_paths = get_filepaths(frames_dir)
scores = []
img_score = 0
match = ''

for img in image_paths:

    similarity_score = compare_images(img, target_image)
    # scores.append(similarity_score)

    if similarity_score > img_score:

        img_score = similarity_score
        match = img

print("Highest similarity is: ", match, "\n", "Similarity is: ", img_score)

