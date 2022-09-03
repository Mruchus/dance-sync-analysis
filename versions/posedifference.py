# Copy from previous version of work

import cv2
import mediapipe as mp
from statistics import mean
import math
import numpy as np
import os


# find duration of clips
def get_duration(filename):
    video = cv2.VideoCapture(filename)

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    duration = (frame_count/fps)
    return duration

# utils
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose

def landmarks(video):

    print("Processing video...")

    #laod video
    pose = mpPose.Pose()
    cap = cv2.VideoCapture(video)

    frames = []
    shots = []
    results = []

    frame_count = int(math.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    #print(f'frame count is {frame_count}')

    # process video
    frame_count = 50
    for i in range(frame_count):
        success, img = cap.read()
        shots.append(img)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(imgRGB)
        results.append(result)

        # information about the joint positions
        frames.append([(lm.x, lm.y, lm.z, lm.visibility) for lm in result.pose_landmarks.landmark])
        #print(i)

    print("Video finished processing...")
    return frames, shots, results

def difference(dl1, dl2, s1, s2, r1, r2):
    # all the joints we are using
    connections = [(16, 14), (14, 12), (12, 11), (11, 13), (13, 15), (12, 24), (11, 23), (24, 23), (24, 26), (23, 25), (26, 28), (25, 27)]
    deduction = 0
    deduction_score = 0

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
            print(f"these are the gradients {g1, g2}")

            # difference
            dif = abs((g1 - g2) / g1)


            # use this as a percentage difference
            #print(f"here is the difference: {dif}")

            shot_percentage.append(abs(dif))

        #print(shot_percentage)
        shot_dif = mean(shot_percentage)
        #print(f"difference per shot {shot_dif}")

        # if the dancers aren't matching with each other
        mpDraw.draw_landmarks(s1[f], r1[f].pose_landmarks, mpPose.POSE_CONNECTIONS)
        mpDraw.draw_landmarks(s2[f], r2[f].pose_landmarks, mpPose.POSE_CONNECTIONS)
        comp = np.concatenate((s1[f], s2[f]), axis=1)

        if shot_dif > 10:
            cv2.putText(comp, "!", (340, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3)
            # we are going to deduct some points of the moves are significantly different
            deduction += 100
        cv2.imshow(str(f), comp)
        cv2.waitKey(1)
        if shot_dif > 10:
            cv2.waitKey(500)


    #print(f"The number of frames is {frames}")
    deduction_score = round(deduction / frames)
    return deduction_score

# ---MAIN---
# processing our two dancers
dancer1, d1_shots, d1_res = landmarks('chuucut.mov')
#chuu, c_shots, c_res = landmarks('chuu.mp4')
dancer2, d2_shots, d2_res = landmarks('yvescut.mov')

deduction_score = difference(dancer1, dancer2, d1_shots, d2_shots, d1_res, d2_res)
print(f"You are {100 - round(deduction_score)}% in sync with Yves!")

