# modules
import os
from glob import glob

# load clip
clip = glob('*mov')[0]
clip_name = clip.split(".")[0] + "24" # new clip name

# main command
command = "ffmpeg -i {0} -filter:v fps=24 {1}.mov".format(clip, clip_name)
os.system(command)









