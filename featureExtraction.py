# Import necessary libraries and functions
from functions import extract_frames, process_subdirectory, get_filepaths, get_matching_files

# Define the directories for frames, label images, and videos
frames_dir = "Frames"
labels_dir = "Label_Images"
videos_dir = "Videos"
# As well as if frames should be extracted, in case there are videos that have not
# been processed yet
extract = False

# Obtain a list of all video file paths within the specified videos directory
video_paths = get_filepaths(videos_dir)  # "Videos" for all paths and not just for a single TestVideo

# Iterate over each video path to extract frames
if extract:
    for path in video_paths:
        # Call the extract_frames function for each video
        # The function extracts every frame from the video,
        # '-1' for start and end parameters indicates processing the entire video
        count = extract_frames(path, frames_dir, overwrite=False, start=-1, end=-1, every=1)
        print(path)

# Find matches between frame subdirectories and label images, obtaining a dictionary of matches
# and a list of the frame subdirectories that have corresponding label images
matches, frames_subdirectories = get_matching_files(frames_dir, labels_dir)

# Loop over each frame subdirectory that has potential matches in the labels directory
for subdir in frames_subdirectories:
    # Process each subdirectory by comparing its images against the label images,
    # and then generate and save a JSON file with the comparison results
    process_subdirectory(subdir, matches, frames_dir, labels_dir)
