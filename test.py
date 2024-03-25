import cv2
import os
from fuzzywuzzy import process


# Source: https://medium.com/@haydenfaulkner/extracting-frames-fast-from-a-video-using-opencv-and-python-73b9b7dc9661
def extract_frames(video_path, frames_dir, overwrite=False, start=-1, end=-1, every=1):
    """
    Extract frames from a video using OpenCVs VideoCapture
    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :return: count of images saved
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

            path = os.path.join(save_path, "{:04d}.jpg".format(frame))  # create the save path

            #  print(path)
            if not os.path.exists(path) or overwrite:  # if it doesn't exist, or we want to overwrite anyway

                # Resize the frame
                resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

                cv2.imwrite(path, resized_image)  # save the extracted image
                saved_count += 1  # increment our counter by one

        frame += 1  # increment our frame count

    capture.release()  # after the while has finished close the capture

    return saved_count  # and return the count of the images we saved


# Source: https://medium.com/@patelharsh7458/python-script-which-compare-two-images-and-determine-if-they-are-the
# -same-even-when-one-of-them-is-ee3c8df2a29b
def compare_images(image1_path, image2_path):
    # Load the two images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    # Initialize the ORB (Oriented FAST and Rotated BRIEF) detector
    orb = cv2.ORB.create(nfeatures=5000)

    # Find keypoints and descriptors in both images
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Create a BFMatcher (Brute Force Matcher)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort them by distance (smaller distances are better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate a match score
    match_score = len(matches)

    # print ("Match Score: ", match_score)

    # You can adjust a threshold here to determine if the images are the same
    # Smaller threshold values may yield more lenient matching.
    # threshold = 453 # Adjust as needed

    # if match_score > threshold:
    #     print("Image Nr.: ", image1_path)
    #     print("Images are similar.")
    # else:
    #     print("Images are not similar.")

    # Visualization (optional)
    # result_image = cv2.drawMatches(image1, kp1, image2, kp2, matches[:10], outImg=None)

    # cv2.imshow("Result Image", result_image)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

    return match_score


def get_filepaths(directory):
    # List to store file paths
    file_paths = []

    # List all files and directories in the specified path
    for filename in os.listdir(directory):
        # Construct full file path
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):  # Check if it's a file and not a directory
            file_paths.append(file_path)

    return file_paths


def rm_file_extension(filename):
    file_wo_extension = os.path.splitext(filename)[0]
    return file_wo_extension


def get_matching_files(path_to_frames, path_to_labels):
    # Define a minimum similarity score threshold (0-100)
    similarity_threshold = 70  # Adjust this value as needed

    # Get a list of subdirectories and filenames
    subdirs = [d for d in os.listdir(path_to_frames) if os.path.isdir(os.path.join(path_to_frames, d))]
    labels = [f for f in os.listdir(path_to_labels) if os.path.isfile(os.path.join(path_to_labels, f))]

    # Attempt to match subdirectories to files
    matches = {}
    for subdir in subdirs:
        # Use fuzzy matching to find all file matches above the threshold
        all_matches = process.extract(subdir, labels, limit=6)

        # Filter matches by the similarity score threshold
        # good_matches = [match for match in all_matches if match[1] >= similarity_threshold]

        # Filter matches by the similarity score threshold and extract only filenames
        good_matches = [match[0] for match in all_matches if match[1] >= similarity_threshold]

        # Print and store the good matches (filenames only)
        if good_matches:
            print(f"Good matches for '{subdir}': {good_matches}")
            matches[subdir] = good_matches
        else:
            print(f"No good matches found for '{subdir}'")

    return matches, subdirs


def get_unmentioned_paths(directory, dictionary):
    # Step 1: List all files in the target directory
    all_files = set()
    for root, dirs, files in os.walk(directory):  # Adjust 'TestFrames' to your target directory
        for file in files:
            # Construct the full path and add it to the set
            full_path = os.path.join(root, file)
            all_files.add(full_path)

    # Step 2: Gather all mentioned paths from your dictionary
    mentioned_paths = set(dictionary.values())

    # Step 3: Find the difference - paths in all_files but not in mentioned_paths
    not_mentioned_paths = list(all_files - mentioned_paths)

    return not_mentioned_paths
