import os
import json
from ultralytics import YOLO
import torch
import pandas as pd
from functions import get_filepaths, calculate_distance_travelled, simplify_filename, shear_video

print(torch.cuda.is_available())

# Load a model
model = YOLO('yolov8x-pose.pt')  # load yolo pose estimation model

frames_dir = "Frames"
videos_dir = "Videos"
# Obtain a list of all video file paths within the specified videos directory
video_paths = get_filepaths(videos_dir)  # "Videos" for all paths and not just for a single TestVideo

print(video_paths)

augment_data = True
augmentation_suffix = "_augmented_4"

for xcs_video in video_paths:

    print("Current video path: ", xcs_video)

    if augment_data:

        simplified_name = simplify_filename(xcs_video)

        simplified_name += augmentation_suffix + ".mp4"
        output_video_path = os.path.join(videos_dir, simplified_name)

        sheared_video_path = shear_video(xcs_video, output_video_path)

        tracking_results = model.track(source=sheared_video_path, conf=0.5, show=False, save=False)  # tracking on video

        try:
            # Remove the file
            os.remove(sheared_video_path)
            print(f"File '{sheared_video_path}' was deleted successfully.")
        except FileNotFoundError:
            print(f"The file '{sheared_video_path}' does not exist.")
        except PermissionError:
            print("Permission denied: You do not have the necessary permissions to delete the file.")
        except OSError as error:
            print(f"Error: {error}")

    else:
        tracking_results = model.track(source=xcs_video, conf=0.5, show=False, save=False)  # tracking on video

    frame_numbers = []
    ids = []
    bboxes = []
    kps = []

    kp_confs = []

    # Assume `video_results` is a list where each element represents the results for a frame
    for frame_index, results in enumerate(tracking_results):
        for r in results:

            # Extract data for the current frame
            box_id = r.boxes.id.numpy() if r.boxes.id is not None else [0]
            bbox = r.boxes.xywh.cpu().numpy()
            kp = r.keypoints.xy.cpu().numpy()
            kp_conf = r.keypoints.conf.cpu().numpy()

            # Append data to lists
            frame_numbers.extend([frame_index] * len(box_id))
            ids.extend(box_id)
            bboxes.extend(bbox)
            kps.extend(kp)
            kp_confs.extend(kp_conf)

    # Create DataFrame
    df = pd.DataFrame({
        'FrameNumber': frame_numbers,
        'ID': ids,
        'BoundingBox': list(map(tuple, bboxes)),  # Convert arrays to tuples for better handling
        'Keypoints': list(map(tuple, kps)),
        'Keypoint Conf': list(map(tuple, kp_confs))
    })

    # Set multi-index for easier data retrieval by frame and ID
    df.set_index(['FrameNumber', 'ID'], inplace=True)

    distances = {}

    # Get unique IDs
    unique_ids = df.index.get_level_values('ID').unique()

    # Loop over each unique ID
    for uid in unique_ids:

        if uid == 0:
            distances[uid] = 0
        else:
            id_data = df.xs(uid, level='ID')
            first_frame = id_data.index.min()
            last_frame = id_data.index.max()

            # Get bounding box data for the first and last appearances
            first_bbox = id_data.loc[first_frame, 'BoundingBox']
            last_bbox = id_data.loc[last_frame, 'BoundingBox']

            # Calculate the distance traveled by the centroid
            distance = calculate_distance_travelled(first_bbox, last_bbox)
            distances[uid] = distance

    # Find the ID with the maximum distance traveled
    max_distance_id = max(distances, key=distances.get)

    print(f"ID {max_distance_id} has traveled the longest distance of {distances[max_distance_id]} units.")

    # Number of keypoints (COCO model has 17 keypoints)
    num_kps = 17

    # Create column names for the DataFrame
    columns = ['Posx', 'Posy', 'width', 'height'] + \
              [f'PosX Kp{i + 1}' for i in range(num_kps)] + \
              [f'PosY Kp{i + 1}' for i in range(num_kps)] + \
              [f'Conf Kp{i + 1}' for i in range(num_kps)] + \
              ['Label']

    # Initialize DataFrame for selected xc skier
    xcs_df = pd.DataFrame(index=range(len(tracking_results)), columns=columns)

    # Get data for selected xc skier from the original DataFrame
    xcs_data = df.xs(max_distance_id, level='ID')

    # Loop through all frames where selected xc skier is detected and fill the DataFrame
    for frame in xcs_data.index:
        bbox = xcs_data.loc[frame, 'BoundingBox']
        kps = xcs_data.loc[frame, 'Keypoints']
        kp_confs = xcs_data.loc[frame, 'Keypoint Conf']

        # Fill bounding box data
        xcs_df.loc[frame, ['Posx', 'Posy', 'width', 'height']] = bbox

        # Fill keypoint data
        for i in range(num_kps):
            x_pos, y_pos = kps[i]

            xcs_df.loc[frame, [f'PosX Kp{i + 1}']] = x_pos
            xcs_df.loc[frame, [f'PosY Kp{i + 1}']] = y_pos
            xcs_df.loc[frame, [f'Conf Kp{i + 1}']] = kp_confs[i]

    # Fill missing data for frames where ID 4 isn't detected
    xcs_df.fillna(0, inplace=True)  # Replace None with 0 or use np.nan to keep as NaN
    xcs_df = xcs_df.infer_objects()

    simplified_name = simplify_filename(xcs_video)

    json_file = os.path.join(frames_dir, simplified_name, 'labels.json')

    with open(json_file, 'r') as file:
        label_info_full = json.load(file)

    # Extract only the first five items
    label_info = dict(list(label_info_full.items())[:5])

    # Process the JSON data and update DataFrame
    for label, path in label_info.items():
        # Extract the frame number from the file path
        frame_number = int(path.split('\\')[-1].replace('.png', ''))

        print(frame_number)

        # Update the label in the DataFrame if the frame number exists in the index
        if frame_number in xcs_df.index:
            xcs_df.loc[frame_number, ['Label']] = int(label)

    if augment_data:
        simplified_name += augmentation_suffix
        # Define the path where you want to create the directory
        directory = os.path.join(frames_dir, simplified_name)

        # Check if the directory already exists
        if not os.path.exists(directory):
            # Create the directory
            os.mkdir(directory)
            print(f"Directory '{directory}' created successfully.")
        else:
            print(f"Directory '{directory}' already exists.")

    xcs_csv_file_path = os.path.join(frames_dir, simplified_name, 'feature_extraction.csv')

    print(xcs_csv_file_path)

    xcs_df.to_csv(xcs_csv_file_path, index=False)

    print("Features extracted and stored to: ", xcs_csv_file_path)
