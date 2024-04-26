import cv2
import os
import numpy as np
import json
from fuzzywuzzy import process
import matplotlib.pyplot as plt


# Source: https://medium.com/@haydenfaulkner/extracting-frames-fast-from-a-video-using-opencv-and-python-73b9b7dc9661
# Reworked with ChatGPT for better performance and error handling
def extract_frames(video_path, frames_dir, overwrite=False, start=-1, end=-1, every=1):
    """
    Extracts frames from a video file and saves them to a directory.

    :param video_path: The path to the video file.
    :param frames_dir: The directory where extracted frames will be saved.
    :param overwrite: If True, overwrite existing frames. Default is False.
    :param start: The starting frame number for extraction. Default is -1 (start from the beginning).
    :param end: The ending frame number for extraction. Default is -1 (go till the end).
    :param every: Extract every nth frame. Default is 1 (extract every frame).
    :return: The count of frames saved.
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    filename = rm_file_extension(video_filename)

    assert os.path.exists(video_path)  # assert the video file exists

    capture = cv2.VideoCapture(video_path)  # open the video using OpenCV

    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.set(1, start)  # set the starting frame of the capture
    frame = start  # keep track of which frame we are up to, starting from start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    saved_count = 0  # a count of how many frames we have saved

    new_width = 1920
    new_height = 1080

    while frame < end:  # let's loop through the frames until the end

        _, image = capture.read()  # read an image from the capture

        if while_safety > 500:  # break the while if our safety maxs out at 500
            break

        # sometimes OpenCV reads None's during a video, in which case we want to just skip
        if image is None:  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            continue  # skip

        if frame % every == 0:  # if this is a frame we want to write out based on the 'every' argument
            while_safety = 0  # reset the safety count

            save_path = os.path.join(frames_dir, filename)

            try:
                os.makedirs(save_path, exist_ok=True)
            except PermissionError:
                print("Permission denied: Unable to create directory due to insufficient permissions.")
            except FileExistsError:
                print("A file with the same name as the directory already exists.")
            except OSError as error:
                print(f"Error creating directory: {error}")

            path = os.path.join(save_path, "{:04d}.png".format(frame))  # create the save path

            #  print(path)
            if not os.path.exists(path) or overwrite:  # if it doesn't exist, or we want to overwrite anyway

                # Resize the frame
                resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

                cv2.imwrite(path, resized_image)  # save the extracted image
                saved_count += 1  # increment our counter by one

        frame += 1  # increment our frame count

    capture.release()  # after the while has finished close the capture

    return saved_count  # and return the count of the images we saved


# Reworked with ChatGPT for better performance
def process_subdirectory(directory_to_go_through, match_dictionary,
                         frames_dir, labels_dir):
    """
    Processes a single subdirectory within frames_dir by comparing its images
    against a set of label images and generating a JSON file with the comparison results.

    :param directory_to_go_through: The name of the subdirectory to process.
    :param match_dictionary: A dictionary with keys as subdirectory names and values as lists of matching label image filenames.
    :param frames_dir: The root directory containing frame subdirectories.
    :param labels_dir: The directory containing label images to compare against the frames.

    If a 'labels.json' file already exists in the subdirectory, the function will skip processing.
    The function writes a 'labels.json' file containing matched image paths and unmentioned paths.
    """

    # Retrieve list of label images that match the current subdirectory
    if directory_to_go_through not in match_dictionary:
        print("No match for " + directory_to_go_through)
        return  # Exit the function if there are no matches.

    # Retrieve the list of label images that match the current subdirectory.
    label_images = match_dictionary[directory_to_go_through]

    # Build the full file path to the current subdirectory of frames.
    path = os.path.join(frames_dir, directory_to_go_through)

    # Get all image file paths within the current subdirectory
    image_paths = get_filepaths(path)  # yields something like 'Frames/1_Petzold Luca_lq/0001.png...110.png'

    # Check if the subdirectory should be skipped because 'labels.json' already exists.
    skip_subdir = any(i.endswith('labels.json') for i in image_paths)
    if skip_subdir:
        print("Skipping subdirectory because labels.json exists.")
        return  # Skip further processing for this subdirectory.

    # If skip_subdir is True, skip the rest of the processing for this subdir
    if skip_subdir:
        return

    # Initialize structures for storing match results and their corresponding scores.
    match_dictionary = {str(i): '' for i in range(1, 6)}
    image_scores = {str(i): 0 for i in range(1, 6)}

    # Map the endings of label filenames to keys in match_dictionary and image_scores.
    label_endings_to_keys = {f'00{i}.jpg': str(i) for i in range(1, 6)}

    # Iterate over each image path within the current subdirectory
    for img in image_paths:
        # Compare the current image with each label image
        for label_filename in label_images:
            # Construct the full path to the current label image
            label = os.path.join(labels_dir, label_filename)
            # Compute the similarity score between the current frame and label image
            similarity_score = compare_images(img, label)

            # Update dictionary and scores if the current score is higher.
            for ending, key in label_endings_to_keys.items():
                if label_filename.endswith(ending) and similarity_score > image_scores[key]:
                    image_scores[key] = similarity_score
                    match_dictionary[key] = img
                    break  # Stop checking once a match is updated.

            print(match_dictionary)

    # Get paths of images that were not matched to any label.
    unmentioned_paths = get_unmentioned_paths(path, match_dictionary)

    # Combine matched paths and unmentioned paths into a structured data for JSON
    # Compile the matching results into a single data structure.
    combined_data = {**match_dictionary, "0": unmentioned_paths}

    # Create the path for the resulting JSON file in the current subdirectory.
    json_filepath = os.path.join(path, "labels.json")

    # Write the match results to the JSON file.
    with open(json_filepath, 'w') as json_file:
        json.dump(combined_data, json_file, indent=4)


# Source: https://medium.com/@patelharsh7458/python-script-which-compare-two-images-and-determine-if
# -they-are-the -same-even-when-one-of-them-is-ee3c8df2a29b
# Reworked with ChatGPT for higher accuracy
def compare_images(image1_path, image2_path):
    """
    Compares two images and calculates a similarity score based on filtered keypoint matching.

    :param image1_path: Path to the first image file.
    :param image2_path: Path to the second image file.
    :return: Number of good matches after filtering and verification.
    """
    # Load images in grayscale
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Initialize ORB detector
    orb = cv2.ORB.create(nfeatures=5000)

    # Find key points and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    # Create BFMatcher object and find the k best matches for each descriptor
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to filter matches
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    # Homography check
    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography matrix and perform RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # draw_keypoints_and_matches(image1, image2, kp1, kp2, good_matches, image1_path, image2_path)

        # Count the number of inliers (matches that fit the homography)
        return np.sum(matchesMask)
    else:
        return 0


# Source of Inspiration: ChatGPT
# This function is used to generate exemplary images for GitHub
def draw_keypoints_and_matches(img1, img2, keypoints1, keypoints2, matches, image1_path, image2_path):
    """
    Draw keypoints on both images and good matches between them using Matplotlib,
    including scaling keypoints for resized images and options to save the comparison.

    :param img1: The first image.
    :param img2: The second image.
    :param keypoints1: Keypoints in the first image.
    :param keypoints2: Keypoints in the second image.
    :param matches: Filtered good matches between keypoints.
    :param image1_path: Path to the first image file for title display.
    :param image2_path: Path to the second image file for title display.
    """
    scale_percent = 39  # Example scale percentage for resizing

    # Resize images
    image1_resized = resize_image(img1, scale_percent)

    # Scale keypoints coordinates
    kp1_scaled = [cv2.KeyPoint(kp.pt[0] * scale_percent / 100, kp.pt[1] * scale_percent / 100, kp.size) for kp in
                  keypoints1]

    # Draw good matches on the resized images with scaled keypoints
    img_matches = cv2.drawMatches(image1_resized, kp1_scaled, img2, keypoints2, matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Convert images to RGB for Matplotlib
    img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)

    # Use Matplotlib to display the images
    fig = plt.figure()
    plt.imshow(img_matches_rgb)

    plt.title(f"Image Comparison: {os.path.split(image1_path)[1]} vs. {os.path.split(image2_path)[1]}")
    plt.yticks([])
    plt.xticks([])

    # Option to save the figure
    save_fig = input("Do you want to save this comparison? (yes/no): ").lower()
    if save_fig == 'yes':
        save_path = os.path.split(image1_path)[0] + "\\" + "comparison_result.svg"
        fig.savefig(save_path, format='svg', dpi=600)
        print(f"Comparison saved to {save_path}")

    plt.show()


def resize_image(image, scale_percent):
    """
    Resizes an image by a specific scale percentage.

    :param image: The image to resize.
    :param scale_percent: The desired scale percentage.
    :return: The resized image.
    """
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


# Source of Inspiration: ChatGPT
def get_filepaths(directory, extension=None):
    """
    Retrieves the file paths of all files in the specified directory,
    optionally filtering by file extension. If only one file is found,
    returns a single string. Otherwise, returns a list of file paths.

    :param directory: The directory from which to list file paths.
    :param extension: Optional. Specify the file extension to filter by (e.g., 'json').
                      The extension should be provided without a dot.
    :return: A single file path as a string if only one file is found,
             or a list of file paths if multiple files are found.
    """

    # List to store file paths
    file_paths = []

    # List all files and directories in the specified path
    for filename in os.listdir(directory):
        # Construct full file path
        file_path = os.path.join(directory, filename)
        # Check if it's a file and not a directory
        if os.path.isfile(file_path):
            # If an extension is specified, filter files by the extension
            if extension and filename.endswith('.' + extension):
                file_paths.append(file_path)
            elif not extension:  # If no extension is specified, add all files
                file_paths.append(file_path)

    # Return a single file path as a string if only one is found, otherwise return the list
    if len(file_paths) == 1:
        return file_paths[0]
    else:
        return file_paths


def rm_file_extension(filename):
    """
    Removes the file extension from a filename.

    :param filename: The filename from which to remove the extension.
    :return: The filename without its extension.
    """
    file_wo_extension = os.path.splitext(filename)[0]
    return file_wo_extension


# Source of Inspiration: ChatGPT
def get_matching_files(path_to_frames, path_to_labels):
    """
    Matches frame subdirectories to label images using fuzzy string matching.

    :param path_to_frames: The directory containing frame subdirectories.
    :param path_to_labels: The directory containing label images.
    :return: A tuple containing a dictionary of matches and a list of frame subdirectories.
    """
    # Define a minimum similarity score threshold (0-100)
    similarity_threshold = 70  # Adjust this value as needed

    # Get a list of subdirectories and filenames
    subdirs = [d for d in os.listdir(path_to_frames) if os.path.isdir(os.path.join(path_to_frames, d))]
    labels = [f for f in os.listdir(path_to_labels) if os.path.isfile(os.path.join(path_to_labels, f))]

    # Attempt to match subdirectories to files
    matches = {}
    for subdir in subdirs:
        # Use fuzzy matching to find all file matches above the threshold
        all_matches = process.extract(subdir, labels, limit=5)

        # Filter matches by the similarity score threshold and extract only filenames
        good_matches = [match[0] for match in all_matches if match[1] >= similarity_threshold]

        # Print and store the good matches (filenames only)
        if good_matches:
            # print(f"Good matches for '{subdir}': {matches}")
            matches[subdir] = good_matches
        else:
            print(f"No good matches found for '{subdir}'")

    return matches, subdirs


# Source of Inspiration: ChatGPT
def get_unmentioned_paths(directory, dictionary):
    """
    Finds file paths in the specified directory that are not mentioned in the given dictionary.

    :param directory: The directory to search within.
    :param dictionary: A dictionary containing mentioned file paths as values.
    :return: A list of file paths not mentioned in the dictionary.
    """
    # Step 1: List all files in the target directory
    all_files = set(get_filepaths(directory))

    print(all_files)

    # Step 2: Gather all mentioned paths from your dictionary
    # Filter out empty strings and normalize paths
    mentioned_paths = set(os.path.normpath(path) for path in dictionary.values() if path)

    # Step 3: Find the difference - paths in all_files but not in mentioned_paths
    not_mentioned_paths = list(all_files - mentioned_paths)

    # Optionally sort the list if you need an ordered result
    not_mentioned_paths.sort()

    return not_mentioned_paths


# Source of Inspiration: ChatGPT
def calculate_distance_travelled(box1, box2):
    """ Calculate Euclidean distance between the centroids of two bounding boxes. """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    centroid1 = (x1 + w1 / 2, y1 + h1 / 2)
    centroid2 = (x2 + w2 / 2, y2 + h2 / 2)
    return np.sqrt((centroid2[0] - centroid1[0]) ** 2 + (centroid2[1] - centroid1[1]) ** 2)


def simplify_filename(filepath):
    # Extract the filename without any directory path
    filename = os.path.basename(filepath)
    # Remove the file extension
    simplified_filename, _ = os.path.splitext(filename)

    return simplified_filename
