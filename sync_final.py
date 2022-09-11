# current version 03/09/22

import os
import subprocess
from glob import glob
import cv2
from cv2 import VideoWriter_fourcc, VideoWriter
import math
import mediapipe as mp
from statistics import mean
import numpy as np

def get_duration(filename): # returns the duration of a clip in seconds
    video = cv2.VideoCapture(filename)

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    duration = (frame_count/fps)
    return duration

def get_frame_count(filename): #returns the number of frames in a clip
    cap = cv2.VideoCapture(filename)

    frame_count = int(math.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    return cap, frame_count

# utils
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose

def landmarks(video):

    #print(f"Processing {video}...")

    #laod video
    pose = mpPose.Pose()

    frames = []
    shots = []
    results = []

    # get the video capture and frame count of video
    cap, frame_count = get_frame_count(video)
    #print(f'frame count is {frame_count}')

    # process video
    for i in range(frame_count):
        success, img = cap.read()
        shots.append(img)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(imgRGB)
        results.append(result)

        # information about the joint positions
        frames.append([(lm.x, lm.y) for lm in result.pose_landmarks.landmark])
        #print(i)

    #print("Video finished processing...")
    return frames, shots, results

def difference(dl1, dl2, s1, s2, r1, r2):
    # all the joints we are using
    connections = [(16, 14), (14, 12), (12, 11), (11, 13), (13, 15), (12, 24), (11, 23), (24, 23), (24, 26), (23, 25), (26, 28), (25, 27)]
    deduction = 0
    outofsyncframe = 0
    live_score = 100

    # number of frames
    frames = min(len(dl1), len(dl2))
    #print(f"We are processing {frames} frames")

    print("Analysing dancers...")

    video = VideoWriter('output.mp4', VideoWriter_fourcc(*'mp4v'), 24.0, (2*720, 1280), isColor=True)

    for f in range(frames):
        # percentage difference of joints per frame
        shot_percentage = []

        # each model for the frame
        p1, p2 = dl1[f], dl2[f]
        for connect in connections:
            m1, m2 = connect

            # find gradients
            #print(p1[m1][1], p1[m2][1], p1[m1][0], p1[m2][0])
            #print(p2[m1][1], p2[m2][1], p2[m1][0], p2[m2][0])
            g1 = (p1[m1][1] - p1[m2][1]) / (p1[m1][0] - p1[m2][0])
            g2 = (p2[m1][1] - p2[m2][1]) / (p2[m1][0] - p2[m2][0])
            #print(f"these are the gradients {g1, g2}")

            # difference
            dif = abs((g1 - g2) / g1)

            # use this as a percentage difference
            #print(f"here is the difference: {dif}")
            shot_percentage.append(abs(dif))

        # FINISHED analysing connections
        shot_dif = mean(shot_percentage)
        #print(f"difference per shot {shot_dif}")

        # drawing
        frame_height, frame_width, _ = s1[f].shape
        mpDraw.draw_landmarks(s1[f], r1[f].pose_landmarks, mpPose.POSE_CONNECTIONS)
        mpDraw.draw_landmarks(s2[f], r2[f].pose_landmarks, mpPose.POSE_CONNECTIONS)
        comp = np.concatenate((s1[f], s2[f]), axis=1)

        colour = (0, 0, 255) if shot_dif > 10 else (255, 0, 0)

        cv2.putText(comp, f"Diff: {shot_dif:.2f}", (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 3)

        # if the dancers aren't matching with each other - display warning
        if shot_dif > 10:
            outofsyncframe += 1 # use for deduction

        live_score = ((f+1 - outofsyncframe) / (f+1)) * 100.0
        cv2.putText(comp, f"Score: {live_score:.2f}%", (frame_width +40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 3)

        cv2.imshow(str(f), comp)
        video.write(comp)
        cv2.waitKey(1) # show frame

    video.release()
    return live_score


# --------------------------------------------------------- SYNCING -----------------------------------------------------------------------------------
# REFERENCING
def convert_to_same_framerate():
    # find all video clips in directory
    clip_list = glob('*mov')
    # first we make sure that all the clips are 24fps
    for clip in clip_list:
        clip_name = clip.split(".")[0] + "24"
        # print(clip_name)
        command = "ffmpeg -i {0} -filter:v fps=24 {1}.mov".format(clip, clip_name)
        os.system(command)

    return clip_name

def choose_reference_clip():
    clip_list = glob('*24.mov')
    # print(clip_list)
    # reference clip is longer than comparison clip
    if get_frame_count(clip_list[0])[1] > get_frame_count(clip_list[1])[1]:
        ref_clip = clip_list[0]
        comparison_clip = clip_list[1]
    else:
        ref_clip = clip_list[1]
        comparison_clip = clip_list[0]
    print(clip_list)

    return ref_clip, comparison_clip

def convert_to_wav():
    # save ref clip
    command = "ffmpeg -i {0} ref.wav".format(ref_clip)
    os.system(command)
    # cut clip
    clip_name = clip.split(".")[0]
    # print(clip_name)
    # extract audio from other vid and save
    command = "ffmpeg -i {0} {1}.wav".format(clip, clip_name)
    os.system(command)

    return clip_name

def find_sound_offset(clip_name):
    # find offset
    command = "/Applications/Praat.app/Contents/MacOS/Praat --run 'crosscorrelate.praat' ref.wav {0}.wav 0 30".format(
        clip_name)
    offset = subprocess.check_output(command, shell=True)

    # delete wav files
    command = "rm *wav"
    os.system(command)

    return offset

def trim_clips():
    # reference clip is longer
    # duration is length of comparison clip
    duration = get_duration(clip)
    cut_names = []
    for result in results:
        cut_name = result[0].split(".")[0] + "cut.mov"
        cut_names.append(str(cut_name))
    clip_start = abs(float(results[1][1]))
    command = "ffmpeg -i {0} -ss {1} -t {2} {3}".format(ref_clip, str(clip_start), str(duration), cut_names[0])
    os.system(command)
    command = "ffmpeg -i {0} -ss {1} -t {2} {3}".format(clip, 0, str(duration), cut_names[1])
    os.system(command)

    return cut_names

# clip_name = convert_to_same_framerate()
# ref_clip, clip = choose_reference_clip()
# clip_name = convert_to_wav()
#
# results = []
# results.append((ref_clip, 0))
#
# offset = find_sound_offset(clip_name)
# results.append((clip, str(offset)[2:-4])) # offset in seconds
# print(f"results{results}")
#
# cut_names = trim_clips()

# --------------------------------------------------------- MAIN PROCESSING --------------------------------------------------------------------------------------

cut_names = ['50fpschuu24cut.mov', 'yves24cut.mov']
path = "/Users/mruchus/sync"

# processing our two dancers
print(f"model: {cut_names[0]}, comparision: {cut_names[1]} \n")
dancer1, dancer1_shots, dancer1_res = landmarks(cut_names[0])
dancer2, dancer2_shots, dancer2_res = landmarks(cut_names[1])

score = difference(dancer1, dancer2, dancer1_shots, dancer2_shots, dancer1_res, dancer2_res)
print(f"\n You are {score:.2f} % in sync with your model dancer!")


def remove_final_videos():
    command = "rm *cut.mov"
    os.system(command)
    command = "rm *24.mov"
    os.system(command)

# remove_final_videos()
