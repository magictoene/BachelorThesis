import os
import json
from test import extract_frames, compare_images, get_filepaths, get_matching_files, get_unmentioned_paths

# frames_dir = "Frames"
frames_dir = "TestFrames"
labels_dir = "TestPictures"

video_paths = get_filepaths("Video")  # "Videos" for all paths

for path in video_paths:
    count = extract_frames(path, frames_dir, overwrite=True, start=-1, end=-1, every=1)

matches, frames_subdirs = get_matching_files(frames_dir, labels_dir)

for subdir in frames_subdirs:

    label_images = matches[subdir]
    print(label_images)

    path = os.path.join(frames_dir, subdir)
    image_paths = get_filepaths(path)  # yields something like 'TestFrames/1_Petzold Luca_lq/0001.jpg...110.jpg'

    match_dictionary = {'1': '',
                        '2': '',
                        '3': '',
                        '4': '',
                        '5': '',
                        '6': ''}

    image_scores = {'1': 0,
                    '2': 0,
                    '3': 0,
                    '4': 0,
                    '5': 0,
                    '6': 0}

    for img in image_paths:

        for label_filename in label_images:

            label = os.path.join(labels_dir, label_filename)

            # print("Image: ", img, " Label: ", label_filename)
            similarity_score = compare_images(img, label)

            if label_filename.endswith('001.jpg'):
                if similarity_score > image_scores['1']:
                    image_scores['1'] = similarity_score
                    match_dictionary['1'] = img

            elif label_filename.endswith('002.jpg'):
                if similarity_score > image_scores['2']:
                    image_scores['2'] = similarity_score
                    match_dictionary['2'] = img

            elif label_filename.endswith('003.jpg'):
                if similarity_score > image_scores['3']:
                    image_scores['3'] = similarity_score
                    match_dictionary['3'] = img

            elif label_filename.endswith('004.jpg'):
                if similarity_score > image_scores['4']:
                    image_scores['4'] = similarity_score
                    match_dictionary['4'] = img

            elif label_filename.endswith('005.jpg'):
                if similarity_score > image_scores['5']:
                    image_scores['5'] = similarity_score
                    match_dictionary['5'] = img

            elif label_filename.endswith('006.jpg'):
                if similarity_score > image_scores['6']:
                    image_scores['6'] = similarity_score
                    match_dictionary['6'] = img

            print(match_dictionary)

            unmentioned_paths = get_unmentioned_paths(frames_dir, match_dictionary)

            # Combine dictionary and list into a new structure for JSON
            combined_data = {
                "1": match_dictionary['1'],
                "2": match_dictionary['2'],
                "3": match_dictionary['3'],
                "4": match_dictionary['4'],
                "5": match_dictionary['5'],
                "6": match_dictionary['6'],
                "0": unmentioned_paths
            }

            json_filepath = os.path.join(path, "labels.json")

            # Dump the combined structure to a JSON file
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
