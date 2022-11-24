# "The Great Refactor" 21/11/22

import os
import subprocess
from glob import glob
import cv2
from cv2 import VideoWriter_fourcc, VideoWriter
import math
import mediapipe as mp
from statistics import mean
import numpy as np

# --------------------------------------------------------- VIDEO PROCESSING --------------------------------------------------------------------------------------


def get_duration(filename): # returns the duration of a clip
    captured_video = cv2.VideoCapture(filename)

    fps = captured_video.get(cv2.CAP_PROP_FPS) # frame rate
    frame_count = captured_video.get(cv2.CAP_PROP_FRAME_COUNT)

    duration = (frame_count/fps) # in secs
    return duration


def get_frame_count(filename): #returns the number of frames in a clip
    captured_video = cv2.VideoCapture(filename)

    frame_count = int(math.floor(captured_video.get(cv2.CAP_PROP_FRAME_COUNT)))
    return captured_video, frame_count

# utils
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose

def landmarks(video):
    pose = mpPose.Pose() # initialise pose object

    xy_landmard_coords = [] # we only care about x and y coords, NOT z
    frames = []
    landmarks = []

    # capture video
    captured_video, frame_count = get_frame_count(video)

    # process video
    for i in range(frame_count): 
        success, image = captured_video.read() # read frames one by one
        frames.append(image)
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # formatting

        # get landmarks
        landmarks_in_frame = pose.process(imgRGB)
        landmarks.append(landmarks_in_frame)

        # information about the joint positions
        xy_landmard_coords.append([(lm.x, lm.y) for lm in landmarks_in_frame.pose_landmarks.landmark])

    return xy_landmard_coords, frames, landmarks



def difference(xy1, xy2, frames1, frames2, landmarks1, landmarks): # x and y positions of joints | frames | landmarks - info including z
    # all the joints we are using
    # ref: https://mediapipe.dev/images/mobile/pose_tracking_full_body_landmarks.png
    connections = [(16, 14), (14, 12), (12, 11), (11, 13), (13, 15), (12, 24), (11, 23), (24, 23), (24, 26), (23, 25), (26, 28), (25, 27)]

    # keep track of current number of out of sync frames (OFS)
    out_of_sync_frames = 0
    score = 100

    # number of frames
    num_of_frames = min(len(xy1), len(xy2)) # avoids empty displays

    print("Analysing dancers...")
    #writing our final video
    video = VideoWriter('output.mp4', VideoWriter_fourcc(*'mp4v'), 24.0, (2*720, 1280), isColor=True)

    for f in range(num_of_frames): # f = frame number

        # percentage difference of joints per frame
        percentage_dif_of_frames = []

        # get position of all joints for frame 1,2,3...etc
        p1, p2 = xy1[f], xy2[f]

        for connect in connections:
            j1, j2 = connect
            
            # gradients
            # [j] tells you the joint no. ,  [0] -> x coord , [1] -> y coord
            g1 = (p1[j1][1] - p1[j2][1]) / (p1[j1][0] - p1[j2][0])
            g2 = (p2[j1][1] - p2[j2][1]) / (p2[j1][0] - p2[j2][0])

            # difference (dancer1 taken as reference gradient)
            dif = abs((g1 - g2) / g1)
            percentage_dif_of_frames.append(abs(dif))

        # FINISHED analysing connections
        frame_dif = mean(percentage_dif_of_frames) # mean difference of all limbs per frame

        # DRAW LIVE COMPARISON
        frame_height, frame_width, _ = frames1[f].shape # dancer1 video is reference size
        mpDraw.draw_landmarks(frames1[f], landmarks1[f].pose_landmarks, mpPose.POSE_CONNECTIONS)
        mpDraw.draw_landmarks(frames2[f], landmarks[f].pose_landmarks, mpPose.POSE_CONNECTIONS)
        display = np.concatenate((frames1[f], frames2[f]), axis=1)

        colour = (0, 0, 255) if frame_dif > 10 else (255, 0, 0) # red = big difference, BAD!

        cv2.putText(display, f"Diff: {frame_dif:.2f}", (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 3)

        # could add warning pause / sign here
        if frame_dif > 10:
            out_of_sync_frames += 1 # use for deduction

        # live score
        score = ((f+1 - out_of_sync_frames) / (f+1)) * 100.0 # +1 to avoid divide by zero on first frame
        cv2.putText(display, f"Score: {score:.2f}%", (frame_width +40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 3)

        cv2.imshow(str(f), display)
        video.write(display)
        cv2.waitKey(1) # show frame

    video.release() # finish capturing output video
    return score


# --------------------------------------------------------- SYNCING -----------------------------------------------------------------------------------

def convert_to_same_framerate(clip_list):
    # find all video clips in directory
    for clip in clip_list:
        # convert to 24fps
        clip_name = clip.split(".")[0] + "24"
        command = "ffmpeg -i {0} -filter:v fps=24 {1}.mov".format(clip, clip_name)
        os.system(command)



def choose_reference_clip(clip_list):
    # IMPORTANT: reference clip is longer than comparison clip
    if get_frame_count(clip_list[0])[1] > get_frame_count(clip_list[1])[1]: # determine ref and comparison
        ref_clip = clip_list[0]
        comparison_clip = clip_list[1]
    else:
        ref_clip = clip_list[1]
        comparison_clip = clip_list[0]

    return ref_clip, comparison_clip


def convert_to_wav(ref_clip, comparison_clip):
    # Convert REFERENCE
    command = "ffmpeg -i {0} ref.wav".format(ref_clip)
    os.system(command)

    # Convert COMPARISON
    # get the comparison clip name for future assignment
    comparison_clip_name = comparison_clip.split(".")[0]
    command = "ffmpeg -i {0} {1}.wav".format(comparison_clip, comparison_clip_name)
    os.system(command)



# IMPORTANT: The input clips might be difference lengths
# -> trim clips so only compare when dancers are doing the same amount / section of the choreo
def find_sound_offset(clip_name):
    # find offset between: ref.wav and clip_name.wav
    command = "/Applications/Praat.app/Contents/MacOS/Praat --run 'crosscorrelate.praat'" \
              " ref.wav {0}.wav 0 30".format(clip_name)
    # note: code in separate praat file
    offset = subprocess.check_output(command, shell=True)

    # delete wav files (used to sync via reference audio)
    command = "rm *wav"
    os.system(command)

    return offset

# --------------------------------------------------------- COMPUTE SYNC ------------------------------------------------------------------------

# to make sure both clips are same length before comparison
def trim_clips(comparison_clip, offset_results):
    # reference clip is longer
    # duration is length of comparison clip
    duration = get_duration(comparison_clip)
    cut_clips_names = [] # get names of clips so we assign when reformatting
    for result in offset_results:
        cut_name = result[0].split(".")[0] + "cut.mov"
        cut_clips_names.append(str(cut_name))

    # ref clip should start at time of the offset
    clip_start = abs(float(offset_results[1][1]))
    print(clip_start)
    # ref
    command = "ffmpeg -i {0} -ss {1} -t {2} {3}".format(ref_clip, str(clip_start), str(duration), cut_clips_names[0])
    os.system(command)
    # comparison
    command = "ffmpeg -i {0} -ss {1} -t {2} {3}".format(comparison_clip, 0, str(duration), cut_clips_names[1])
    os.system(command)

    return cut_clips_names


def remove_final_videos():
    command = "rm *cut.mov"
    os.system(command)
    command = "rm *24.mov"
    os.system(command)

# --------------------------------------------------------- PREPARE VIDEOS --------------------------------------------------------------------------------------

# adjust frame rate
clip_list = glob('*mov')
print(f"intial clips {clip_list}")
convert_to_same_framerate(clip_list)

# get reference clip
ref_clip, comparison_clip = choose_reference_clip(clip_list)
comparison_clip_name = comparison_clip.split(".")[0] # save the name of comparison clip for future use
print(f'this is the ref: {ref_clip} and comp: {comparison_clip}')

# convert to wav for audio analysis
convert_to_wav(ref_clip, comparison_clip)

# set up offset results & find offset for comparison clip
offset_results = []
offset_results.append((ref_clip, 0)) # one of the clips has no offset

offset = find_sound_offset(comparison_clip_name)
# gets no. secs the comp clip is ahead of the ref clip
offset_results.append((comparison_clip, str(offset)[2:-3])) # offset in seconds
# (did some formatting here to get the offset from b'0.23464366914074475\n to 0.23464366914074475)

print(f"What the heck:{comparison_clip_name} and the results {offset_results}")

final_comparison_clips = trim_clips(comparison_clip, offset_results)

# --------------------------------------------------------- MAIN --------------------------------------------------------------------------------------


# processing our two dancers
print(f"model: {final_comparison_clips[0]}, comparision: {final_comparison_clips[1]} \n")
xy_dancer1, dancer1_frames, dancer1_landmarks = landmarks(final_comparison_clips[0])
xy_dancer2, dancer2_frames, dancer2_landmarks = landmarks(final_comparison_clips[1])

score = difference(xy_dancer1, xy_dancer2, dancer1_frames, dancer2_frames, dancer1_landmarks, dancer2_landmarks)
print(f"\n You are {score:.2f} % in sync with your model dancer!")


remove_final_videos()
