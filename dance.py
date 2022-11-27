import os
import subprocess
from glob import glob
import cv2
from cv2 import VideoWriter_fourcc, VideoWriter
import math
import mediapipe as mp
from statistics import mean
import numpy as np

OUTPUT_DIR="output"
EXIST_FLAG="-n" # ignore existing file, change to -y to always overwrite
PRAAT_PATH="/Applications/Praat.app/Contents/MacOS/Praat"
SEARCH_INTERVAL = 30 # in secs

# --------------------------------------------------------- VIDEO PROCESSING --------------------------------------------------------------------------------------


def get_duration(filename): # returns the duration of a clip
    captured_video = cv2.VideoCapture(filename)

    fps = captured_video.get(cv2.CAP_PROP_FPS) # frame rate
    frame_count = captured_video.get(cv2.CAP_PROP_FRAME_COUNT)

    duration = (frame_count/fps) # in secs
    return duration


def get_frame_count(filename): #returns the number of frames in a clip
    "return tuple of (captured video, frame count)"
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
    video = VideoWriter(f'{OUTPUT_DIR}/output.mp4', VideoWriter_fourcc(*'mp4v'), 24.0, (2*720, 1280), isColor=True)

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


def extract_clip_name(path):
    "extract file name from the path, excluding file extension"
    return path.split('/')[-1].split(".")[0]

def convert_to_same_framerate(clip):
    "convert to 24p and return path to clip with 24fps"
    clip_24 = f"{OUTPUT_DIR}/{extract_clip_name(clip) + '_24'}.mov"
    os.system(f"ffmpeg {EXIST_FLAG} -i {clip} -filter:v fps=24 {clip_24}")
    return clip_24
    
 

def validate_reference_clip(ref_clip, comparison_clip):
    "validate reference clip is longer than comparison clip"
    _, ref_clip_frame_count = get_frame_count(ref_clip)
    _, comparision_clip_frame_count = get_frame_count(comparison_clip)
    if not (ref_clip_frame_count > comparision_clip_frame_count): 
        print(f"Reference clip {ref_clip} has to be longer than comparision clip {comparison_clip}")
        sys.exit(-1)


def convert_to_wav(clip):
    "returns path to wav file of clip"

    clip_wav = f"{OUTPUT_DIR}/{extract_clip_name(clip)}.wav"
    command = f"ffmpeg {EXIST_FLAG} -i {clip} {clip_wav}"
    os.system(command)

    return clip_wav



# IMPORTANT: The input clips might be difference lengths
# -> trim clips so only compare when dancers are doing the same amount / section of the choreo
def find_sound_offset(ref_wav, comparison_wav):
    # find offset between: ref.wav and clip_name.wav
    start_position = 0
    command = f"{PRAAT_PATH} --run 'crosscorrelate.praat' {ref_wav} {comparison_wav} {start_position} {SEARCH_INTERVAL}"
    # note: code in separate praat file
    offset = subprocess.check_output(command, shell=True)
    # (did some formatting here to get the offset from b'0.23464366914074475\n' to 0.23464366914074475)
    # print(f"OFFSET={offset}")
    return abs(float(str(offset)[2:-3]))

# --------------------------------------------------------- COMPUTE SYNC ------------------------------------------------------------------------

# to make sure both clips are same length before comparison
def trim_clips(ref_clip, comparison_clip, offset):
    # duration in secs 
    duration = get_duration(comparison_clip)

    ref_cut = f"{OUTPUT_DIR}/{extract_clip_name(ref_clip) + '_cut.mov'}"
    comparison_cut = f"{OUTPUT_DIR}/{extract_clip_name(comparison_clip) + '_cut.mov'}"

    command = f"ffmpeg {EXIST_FLAG} -i {ref_clip} -ss {offset} -t {duration} {ref_cut}"
    os.system(command)
    command = f"ffmpeg {EXIST_FLAG} -i {comparison_clip} -ss 0 -t {duration} {comparison_cut}"
    os.system(command)

    return ref_cut, comparison_cut


def remove_final_videos():
    command = "rm *cut.mov"
    os.system(command)
    command = "rm *24.mov"
    os.system(command)

# --------------------------------------------------------- PREPARE VIDEOS --------------------------------------------------------------------------------------

import sys
# Launch with these arguments
# python dance.py video/chuu.mov video/cyves.mov

#create output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

if len(sys.argv) < 3:
    print(f"Usage:\n {sys.argv[0]} <ref_clip> <comparison_clip>")
    sys.exit(-1)

ref_clip = sys.argv[1]
comparison_clip = sys.argv[2]

# FULL COMPARE
if not (len(sys.argv)>3 and sys.argv[3] == '--compare-only'):
    print(f"intial clips {ref_clip} {comparison_clip}")
    ref_clip_24, comparison_clip_24 = convert_to_same_framerate(ref_clip), convert_to_same_framerate(comparison_clip)

    # validate reference clip
    print(f'this is the ref: {ref_clip} and comp: {comparison_clip}')
    validate_reference_clip(ref_clip, comparison_clip)


    # # convert to wav for audio analysis
    ref_clip_wav, comparison_clip_wav = convert_to_wav(ref_clip), convert_to_wav(comparison_clip)

    offset = find_sound_offset(ref_clip_wav, comparison_clip_wav)
    # gets no. secs the comp clip is ahead of the ref clip

    ref_cut, comparison_cut = trim_clips(ref_clip_24, comparison_clip_24, offset)
    print(ref_cut, comparison_cut)
# # --------------------------------------------------------- MAIN --------------------------------------------------------------------------------------
else:
    ref_cut = sys.argv[1]
    comparison_cut = sys.argv[2]
    
# processing our two dancers
print(f"model: {ref_cut}, comparision: {comparison_cut} \n")
xy_dancer1, dancer1_frames, dancer1_landmarks = landmarks(ref_cut)
xy_dancer2, dancer2_frames, dancer2_landmarks = landmarks(comparison_cut)

score = difference(xy_dancer1, xy_dancer2, dancer1_frames, dancer2_frames, dancer1_landmarks, dancer2_landmarks)
print(f"\n You are {score:.2f} % in sync with your model dancer!")


# remove_final_videos()
