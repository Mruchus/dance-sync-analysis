import os
from glob import glob

# find all video clips in directory
clip_list = glob('*mov')
for clip in clip_list:
    clip_name = clip.split(".")[0]
    print(clip_name)
    # extract audio from each one and save under the same name with the extension *wav
    command = "ffmpeg -i {0} {1}.wav".format(clip, clip_name)
    os.system(command)


