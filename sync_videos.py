import os
import subprocess
from glob import glob
import cv2
import math
import mediapipe as mp
from statistics import mean
import numpy as np

# find duration of clips
def get_duration(filename):
    video = cv2.VideoCapture(filename)

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    duration = (frame_count/fps)
    return duration


mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose

def landmarks(video):

    print(f"Processing {video}...")

    #laod video
    pose = mpPose.Pose()
    cap = cv2.VideoCapture(video)

    frames = []
    shots = []
    results = []

    frame_count = int(math.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print(f'frame count is {frame_count}')

    # process video
    for i in range(frame_count):
        success, img = cap.read()
        shots.append(img)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(imgRGB)
        results.append(result)

        # information about the joint positions
        frames.append([(lm.x, lm.y) for lm in result.pose_landmarks.landmark])

    print("Video finished processing...")

    return frames, shots, results

def difference(dl1, dl2, s1, s2, r1, r2):
    # all the joints we are goina use
    connections = [(16, 14), (14, 12), (12, 11), (11, 13), (13, 15), (12, 24), (11, 23), (24, 23), (24, 26), (23, 25), (26, 28), (25, 27)]
    # percentage difference for each frame
    percentages = []
    outofsyncshots = []

    # number of frames
    frames = min(len(dl1), len(dl2))

    print("Analysing dancers...")

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

            # difference
            dif = abs((g1 - g2) / g1)

            shot_percentage.append(abs(dif))

        shot_dif = mean(shot_percentage)

        # if the dancers aren't matching with each other
        mpDraw.draw_landmarks(s1[f], r1[f].pose_landmarks, mpPose.POSE_CONNECTIONS)
        mpDraw.draw_landmarks(s2[f], r2[f].pose_landmarks, mpPose.POSE_CONNECTIONS)
        comp = np.concatenate((s1[f], s2[f]), axis=1)
        if shot_dif > 10:
            cv2.putText(comp, "!", (340, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3)
            outofsyncshots.append(comp)
        cv2.imshow(str(f), comp)
        cv2.waitKey(1)
        if shot_dif > 10:
            cv2.waitKey(100)

        percentages.append(shot_dif)

    return mean(percentages), outofsyncshots


# --------------------------------------------------------- SYNCING -----------------------------------------------------------------------------------


# REFERENCING
# find all video clips in directory
clip_list = glob('*mov')
ref_clip = clip_list[0]
clip = clip_list[1]

# save ref clip
command = "ffmpeg -i {0} ref.wav".format(ref_clip)
os.system(command)

results = []
results.append((ref_clip, 0))

# cut clip
clip_name = clip.split(".")[0]
#print(clip_name)

# extract audio from other vid and save
command = "ffmpeg -i {0} {1}.wav".format(clip, clip_name)
os.system(command)

# command to find offset
command = "/Applications/Praat.app/Contents/MacOS/Praat --run 'crosscorrelate.praat' ref.wav {0}.wav 0 30".format(clip_name)
offset = subprocess.check_output(command, shell=True)
# format and save result
results.append((clip, str(offset)[2:-4]))
print(f"results{results}")

# delete wav files
command = "rm *wav"
os.system(command)

#trim clips
# first clip should be longer
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

# --------------------------------------------------------- MAIN PROCESSING --------------------------------------------------------------------------------------

path = "/Users/mruchus/sync"

# processing our two dancers
print(f"clips we are comparing are: {cut_names[0], cut_names[1]}")
dancer1, dancer1_shots, dancer1_res = landmarks(cut_names[0])
dancer2, dancer2_shots, dancer2_res = landmarks(cut_names[1])

dif, outofsyncshots = difference(dancer1, dancer2, dancer1_shots, dancer2_shots, dancer1_res, dancer2_res)
print(f"You are {100 - round(dif)}% in sync with your model dancer!")

#command = "rm *cut.mov"
#os.system(command)
