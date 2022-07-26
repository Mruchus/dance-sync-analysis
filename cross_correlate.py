import os
import subprocess
from glob import glob
import cv2
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
print(clip_name)
# extract audio from other vid and save
command = "ffmpeg -i {0} {1}.wav".format(clip, clip_name)
os.system(command)

# command to find offset
command = "/Applications/Praat.app/Contents/MacOS/Praat --run 'crosscorrelate.praat' ref.wav {0}.wav 0 30".format(clip_name)
offset = subprocess.check_output(command, shell=True)
# format and save result
results.append((clip, str(offset)[2:-4]))
print(results)

# delete wav files
command = "rm *wav"
os.system(command)

#trim clips
duration = get_duration(clip) - float(offset)

cut_names = []
for result in results:
    offset = float(result[1])
    print(result[0].split(".") )
    cut_name = result[0].split(".")[0] + "cut.mp4"
    cut_names += cut_name
    clip_start = offset

    command = "ffmpeg -i {0} -ss {1} -t {2} {3}".format(result[0],
                                                        str(clip_start),
                                                        str(duration),
                                                        cut_name)
    os.system(command)






