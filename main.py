import os
import json
from test import extract_frames, compare_images, get_filepaths, get_matching_files, get_unmentioned_paths

# frames_dir = "Frames"
# labels_dir = "Pictures"
frames_dir = "TestFrames"
labels_dir = "TestPictures"

video_paths = get_filepaths("Video")  # "Videos" for all paths and not just for a single Video

# Iterate over each video path to extract frames
for path in video_paths:
    # Extract frames from each video file, allowing overwrite of existing frames
    # Start and end at default values (-1) to process the entire video, extracting every frame
    count = extract_frames(path, frames_dir, overwrite=True, start=-1, end=-1, every=1)

# Get matching label images for frame subdirectories and a list of those subdirectories
matches, frames_subdirs = get_matching_files(frames_dir, labels_dir)

# Process each frame subdirectory
for subdir in frames_subdirs:
    # Retrieve list of label images that match the current subdirectory
    label_images = matches[subdir]
    print(label_images)

    # Construct the full path to the current subdirectory within frames_dir
    path = os.path.join(frames_dir, subdir)

    # Get all image file paths within the current subdirectory
    image_paths = get_filepaths(path)  # yields something like 'TestFrames/1_Petzold Luca_lq/0001.jpg...110.jpg'

    # Initialize dictionaries to hold match results and similarity scores
    match_dictionary = {'1': '', '2': '', '3': '', '4': '', '5': '', '6': ''}
    image_scores = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0}

    # Define a mapping of label endings to their corresponding keys in match_dictionary and image_scores
    label_endings_to_keys = {
        '001.jpg': '1',
        '002.jpg': '2',
        '003.jpg': '3',
        '004.jpg': '4',
        '005.jpg': '5',
        '006.jpg': '6',
    }

    # Iterate over each image path within the current subdirectory
    for img in image_paths:
        # Compare the current image with each label image
        for label_filename in label_images:
            # Construct the full path to the current label image
            label = os.path.join(labels_dir, label_filename)

            # print("Image: ", img, " Label: ", label_filename)

            # Compute the similarity score between the current frame and label image
            similarity_score = compare_images(img, label)

            # Determine the label key based on the ending of the label_filename
            for ending, key in label_endings_to_keys.items():
                if label_filename.endswith(ending):
                    # Update the match dictionary and image scores if this score is higher
                    if similarity_score > image_scores[key]:
                        image_scores[key] = similarity_score
                        match_dictionary[key] = img
                    break  # Once a match is found, no need to check other endings

            print(match_dictionary)

    # After comparing all images, get paths not mentioned in the match_dictionary
    unmentioned_paths = get_unmentioned_paths(frames_dir, match_dictionary)

    # Combine matched paths and unmentioned paths into a structured data for JSON
    combined_data = {
        "1": match_dictionary['1'],
        "2": match_dictionary['2'],
        "3": match_dictionary['3'],
        "4": match_dictionary['4'],
        "5": match_dictionary['5'],
        "6": match_dictionary['6'],
        "0": unmentioned_paths  # Include unmentioned paths under key "0"
    }

    # Determine the filepath for the JSON file to be saved in the current subdirectory
    json_filepath = os.path.join(path, "labels.json")

    # Write the combined data to the JSON file, formatted with an indentation of 4 spaces for readability
    with open(json_filepath, 'w') as json_file:
        json.dump(combined_data, json_file, indent=4)

# # Example usage:

################################################## Single Video/Image Test #############################################
# target_image = "TestPictures/1_Petzold Luca_001.jpg"
# frames_dir = "TestFrames"
#
# video_paths = get_filepaths("Video")  # "Videos" for all paths
# #
# for path in video_paths:
#     count = extract_frames(path, frames_dir, overwrite=True, start=-1, end=-1, every=1)
#
# frames_dir = "TestFrames/1_Petzold Luca_lq"
# image_paths = get_filepaths(frames_dir)
# scores = []
# img_score = 0
# match = ''
#
# for img in image_paths:
#
#     similarity_score = compare_images(img, target_image)
#     # scores.append(similarity_score)
#
#     if similarity_score > img_score:
#
#         img_score = similarity_score
#         match = img
#
# print("Highest similarity is: ", match, "\n", "Similarity is: ", img_score)
